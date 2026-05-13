import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_REPO = "HuggingFaceTB/SmolLM2-135M"

PROMPTS = [
    "The holographic principle suggests that information in a volume can be described by",
    "Write a Python function that returns the nth Fibonacci number using iteration.",
    "A careful compression experiment should preserve behavior by measuring",
    "In a tiny language model, the attention heads often specialize in",
    "If Alice has three apples and gives Bob one, then Alice has",
    "The capital of France is",
    "Complete the pattern: red, orange, yellow, green,",
    "Explain why quantization error can accumulate across transformer layers.",
]


def quantize_symmetric_group(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("Expected 2D weight tensor")
    if bits < 2:
        raise ValueError("Use at least 2 bits")

    rows, cols = weight.shape
    qmax = (1 << (bits - 1)) - 1
    pad = (group_size - (cols % group_size)) % group_size
    padded = F.pad(weight, (0, pad)) if pad else weight
    grouped = padded.reshape(rows, -1, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(grouped / scale).clamp(-qmax, qmax)
    return (q * scale).reshape(rows, cols + pad)[:, :cols].contiguous()


def quantize_rows_mixed_bits(
    weight: torch.Tensor,
    row_bits: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    output = torch.empty_like(weight)
    for bits in sorted(set(int(v) for v in row_bits.tolist())):
        mask = row_bits == bits
        output[mask] = quantize_symmetric_group(weight[mask], bits, group_size)
    return output


def hadamard(dim: int) -> torch.Tensor:
    if dim <= 0 or dim & (dim - 1):
        raise ValueError("Hadamard requires a power-of-two dimension")
    h = torch.ones(1, 1)
    while h.shape[0] < dim:
        h = torch.cat(
            [
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ],
            dim=0,
        )
    return h / math.sqrt(dim)


def random_sign_perm(dim: int, generator: torch.Generator) -> torch.Tensor:
    perm = torch.randperm(dim, generator=generator)
    signs = torch.randint(0, 2, (dim,), generator=generator, dtype=torch.float32)
    signs = signs.mul(2).sub(1)
    matrix = torch.zeros(dim, dim)
    matrix[torch.arange(dim), perm] = signs
    return matrix


def random_orthogonal(dim: int, generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn(dim, dim, generator=generator)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    return q * signs


def tokenize(tokenizer, prompts: list[str], max_length: int):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def next_token_metrics(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> dict:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().bool()

    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    masked_loss = losses[shift_mask].mean()
    return {
        "loss": float(masked_loss.item()),
        "perplexity": float(torch.exp(masked_loss).item()),
    }


def compare_logits(
    baseline_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    topk: int,
) -> dict:
    mask = attention_mask[:, 1:].bool()
    base = baseline_logits[:, :-1, :]
    cand = candidate_logits[:, :-1, :]
    base_masked = base[mask]
    cand_masked = cand[mask]

    base_logp = F.log_softmax(base_masked, dim=-1)
    cand_logp = F.log_softmax(cand_masked, dim=-1)
    base_p = base_logp.exp()
    kl = F.kl_div(cand_logp, base_p, reduction="batchmean", log_target=False)
    mse = torch.mean((base_masked - cand_masked) ** 2)

    _, base_top = torch.topk(base_masked, k=topk, dim=-1)
    _, cand_top = torch.topk(cand_masked, k=topk, dim=-1)
    overlap = []
    for left, right in zip(base_top, cand_top):
        overlap.append(len(set(left.tolist()) & set(right.tolist())) / topk)

    return {
        "logit_mse": float(mse.item()),
        "kl_from_baseline": float(kl.item()),
        "topk_overlap": float(sum(overlap) / len(overlap)),
    }


def get_layer(model, layer: int):
    return model.model.layers[layer]


def get_attention_shape(model) -> tuple[int, int, int, int]:
    config = model.config
    hidden_size = int(config.hidden_size)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = hidden_size // num_heads
    return hidden_size, num_heads, num_kv_heads, head_dim


def capture_linear_inputs(model, module, batch) -> torch.Tensor:
    captured = []

    def hook(_module, inputs, _output):
        captured.append(inputs[0].detach().float().reshape(-1, inputs[0].shape[-1]).cpu())

    handle = module.register_forward_hook(hook)
    with torch.no_grad():
        model(**batch)
    handle.remove()
    return torch.cat(captured, dim=0)


def evaluate_model(model, batch, baseline_logits=None, topk=10) -> dict:
    with torch.no_grad():
        logits = model(**batch).logits.detach().float().cpu()
    labels = batch["input_ids"].cpu()
    mask = batch["attention_mask"].cpu()
    metrics = next_token_metrics(logits, labels, mask)
    if baseline_logits is not None:
        metrics.update(compare_logits(baseline_logits, logits, mask, topk))
    return metrics | {"logits": logits}


def clone_weight(parameter: torch.nn.Parameter) -> torch.Tensor:
    return parameter.detach().float().cpu().clone()


def assign_weight(parameter: torch.nn.Parameter, value: torch.Tensor) -> None:
    parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def restore_weights(saved: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
    for parameter, value in saved:
        assign_weight(parameter, value)


def apply_value_o_edit_with_matrix(
    model,
    layer_idx: int,
    kv_head: int,
    bits: int,
    group_size: int,
    rot: torch.Tensor,
) -> list:
    layer = get_layer(model, layer_idx)
    v_weight = clone_weight(layer.self_attn.v_proj.weight)
    o_weight = clone_weight(layer.self_attn.o_proj.weight)
    saved = [
        (layer.self_attn.v_proj.weight, v_weight),
        (layer.self_attn.o_proj.weight, o_weight),
    ]

    hidden_size, num_heads, num_kv_heads, head_dim = get_attention_shape(model)
    if kv_head < 0 or kv_head >= num_kv_heads:
        raise ValueError(f"kv_head must be in [0, {num_kv_heads})")
    heads_per_kv = num_heads // num_kv_heads

    kv_start = kv_head * head_dim
    kv_stop = kv_start + head_dim
    group_start = kv_head * heads_per_kv

    v_head = v_weight[kv_start:kv_stop, :]
    o_segments = []
    o_ranges = []
    for query_head in range(group_start, group_start + heads_per_kv):
        col_start = query_head * head_dim
        col_stop = col_start + head_dim
        o_ranges.append((col_start, col_stop))
        o_segments.append(o_weight[:, col_start:col_stop])

    v_transformed = rot.T @ v_head
    o_transformed = [segment @ rot for segment in o_segments]
    v_quant = quantize_symmetric_group(v_transformed, bits, group_size)
    o_quant = [quantize_symmetric_group(segment, bits, group_size) for segment in o_transformed]

    new_v = v_weight.clone()
    new_o = o_weight.clone()
    new_v[kv_start:kv_stop, :] = rot @ v_quant
    for (col_start, col_stop), segment in zip(o_ranges, o_quant):
        new_o[:, col_start:col_stop] = segment @ rot.T

    assign_weight(layer.self_attn.v_proj.weight, new_v)
    assign_weight(layer.self_attn.o_proj.weight, new_o)
    return saved


def value_o_rotation_matrix(head_dim: int, rotation: str, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if rotation == "identity":
        return torch.eye(head_dim)
    if rotation == "hadamard":
        return hadamard(head_dim)
    if rotation == "sign_perm":
        return random_sign_perm(head_dim, generator)
    if rotation == "hadamard_sign_perm":
        return hadamard(head_dim) @ random_sign_perm(head_dim, generator)
    if rotation == "random_orthogonal":
        return random_orthogonal(head_dim, generator)
    raise ValueError(f"Unknown rotation: {rotation}")


def apply_value_o_edit(
    model,
    layer_idx: int,
    kv_head: int,
    bits: int,
    group_size: int,
    rotation: str,
    seed: int = 0,
) -> list:
    _, _, _, head_dim = get_attention_shape(model)
    rot = value_o_rotation_matrix(head_dim, rotation, seed)
    return apply_value_o_edit_with_matrix(model, layer_idx, kv_head, bits, group_size, rot)


def apply_vproj_typical_edit(
    model,
    layer_idx: int,
    bits_hi: int,
    bits_lo: int,
    top_frac: float,
    group_size: int,
    activations: torch.Tensor,
) -> tuple[list, dict]:
    layer = get_layer(model, layer_idx)
    original = clone_weight(layer.self_attn.v_proj.weight)
    saved = [(layer.self_attn.v_proj.weight, original)]

    centered = activations - activations.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    basis = eigenvectors[:, order].contiguous()
    weight_basis = original @ basis

    hidden = original.shape[1]
    top_k = max(1, min(hidden, int(round(hidden * top_frac))))
    col_bits = torch.full((hidden,), bits_lo, dtype=torch.long)
    col_bits[:top_k] = bits_hi

    quant_basis = torch.empty_like(weight_basis)
    for bits in sorted(set(int(v) for v in col_bits.tolist())):
        cols = col_bits == bits
        transposed = weight_basis[:, cols].T.contiguous()
        quant_basis[:, cols] = quantize_symmetric_group(transposed, bits, group_size).T

    reconstructed = quant_basis @ basis.T
    assign_weight(layer.self_attn.v_proj.weight, reconstructed)

    energy = eigenvalues
    retained = float((energy[:top_k].sum() / energy.sum().clamp_min(1e-12)).item())
    effective_bits = (top_k * bits_hi + (hidden - top_k) * bits_lo) / hidden
    info = {
        "top_k": top_k,
        "top_frac": top_frac,
        "retained_activation_energy": retained,
        "effective_bits_per_input_direction": effective_bits,
    }
    return saved, info


def apply_vproj_uniform_edit(model, layer_idx: int, bits: int, group_size: int) -> list:
    layer = get_layer(model, layer_idx)
    original = clone_weight(layer.self_attn.v_proj.weight)
    saved = [(layer.self_attn.v_proj.weight, original)]
    quant = quantize_symmetric_group(original, bits, group_size)
    assign_weight(layer.self_attn.v_proj.weight, quant)
    return saved


def run_probe(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(
        args.repo,
        dtype=torch.float32,
    )
    model.eval()

    prompts = PROMPTS if args.prompts is None else Path(args.prompts).read_text(encoding="utf-8").splitlines()
    prompts = [prompt for prompt in prompts if prompt.strip()]
    batch = tokenize(tokenizer, prompts, args.max_length)

    baseline = evaluate_model(model, batch, topk=args.topk)
    baseline_logits = baseline.pop("logits")

    results = {
        "repo": args.repo,
        "layer": args.layer,
        "bits": args.bits,
        "group_size": args.group_size,
        "prompt_count": len(prompts),
        "baseline": baseline,
        "candidates": [],
    }

    if args.experiment in {"value_o", "both"}:
        for rotation in ["identity", "hadamard"]:
            saved = apply_value_o_edit(
                model,
                layer_idx=args.layer,
                kv_head=args.kv_head,
                bits=args.bits,
                group_size=args.group_size,
                rotation=rotation,
            )
            metrics = evaluate_model(model, batch, baseline_logits, args.topk)
            metrics.pop("logits")
            results["candidates"].append(
                {
                    "experiment": "value_o",
                    "rotation": rotation,
                    "kv_head": args.kv_head,
                    **metrics,
                }
            )
            restore_weights(saved)

    if args.experiment in {"typical_vproj", "both"}:
        layer = get_layer(model, args.layer)
        activations = capture_linear_inputs(model, layer.self_attn.v_proj, batch)

        saved = apply_vproj_uniform_edit(model, args.layer, args.bits, args.group_size)
        metrics = evaluate_model(model, batch, baseline_logits, args.topk)
        metrics.pop("logits")
        results["candidates"].append(
            {
                "experiment": "typical_vproj",
                "candidate": f"uniform_{args.bits}bit",
                **metrics,
            }
        )
        restore_weights(saved)

        saved, info = apply_vproj_typical_edit(
            model,
            layer_idx=args.layer,
            bits_hi=args.typical_hi_bits,
            bits_lo=args.typical_lo_bits,
            top_frac=args.typical_top_frac,
            group_size=args.group_size,
            activations=activations,
        )
        metrics = evaluate_model(model, batch, baseline_logits, args.topk)
        metrics.pop("logits")
        results["candidates"].append(
            {
                "experiment": "typical_vproj",
                "candidate": "pca_typical_mixed_bits",
                **info,
                **metrics,
            }
        )
        restore_weights(saved)

    return results


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_{payload['layer']}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-model SmolLM 135M compression probes")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--experiment", choices=["value_o", "typical_vproj", "both"], default="both")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--kv-head", type=int, default=0)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--typical-hi-bits", type=int, default=4)
    parser.add_argument("--typical-lo-bits", type=int, default=2)
    parser.add_argument("--typical-top-frac", type=float, default=0.25)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--prompts")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    result = run_probe(args)
    path = save_result(result, args.out_dir)
    print(json.dumps(result, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
