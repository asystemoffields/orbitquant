import argparse
import json
import time
import types
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward

from smol135_full_probe import DEFAULT_REPO, evaluate_model, get_attention_shape, get_layer, tokenize
from smol135_kv_rotation_probe import rotation_matrix as kv_rotation_matrix
from smol135_mlp_stack_probe import calibrate_layer_params, patch_mlp_layers, restore_forwards, summarize_candidate
from smol135_sweep import CALIBRATION_PROMPTS, EVAL_PROMPTS, parse_int_list


def load_prompts(path: str | None, fallback: list[str]) -> list[str]:
    if path is None:
        return fallback
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def quantize_last_dim(x: torch.Tensor, bits: int, rotation: torch.Tensor | None, alpha: float = 1.0) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    if rotation is None:
        rotated = x
    else:
        rotation = rotation.to(device=x.device, dtype=x.dtype)
        rotated = x @ rotation
    scale = rotated.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) * alpha / qmax
    quantized = torch.round(rotated / scale).clamp(-qmax, qmax) * scale
    if rotation is None:
        return quantized
    return quantized @ rotation.T


def make_attention_forward(attn, bits: int, rotation_name: str, alpha: float, seed: int):
    head_dim = attn.head_dim
    if rotation_name == "identity":
        rotation = None
    else:
        rotation = kv_rotation_matrix(head_dim, rotation_name, seed)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = quantize_last_dim(key_states, bits, rotation, alpha)
        value_states = quantize_last_dim(value_states, bits, rotation, alpha)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return types.MethodType(forward, attn)


def patch_attention_layers(
    model,
    layers: list[int],
    bits: int,
    rotation_name: str,
    alpha: float,
    seed: int,
) -> list:
    saved = []
    for layer_idx in layers:
        attn = get_layer(model, layer_idx).self_attn
        original_forward = attn.forward
        attn.forward = make_attention_forward(
            attn,
            bits=bits,
            rotation_name=rotation_name,
            alpha=alpha,
            seed=seed + layer_idx * 1009 + len(rotation_name),
        )
        saved.append((attn, original_forward))
    return saved


def run_candidate(
    model,
    eval_batch,
    baseline_logits,
    baseline,
    topk,
    kv_layers,
    kv_bits,
    kv_rotation,
    kv_alpha,
    mlp_params,
    mlp_bits,
    mlp_block_size,
    seed,
) -> tuple[dict, list]:
    saved_attn = []
    saved_mlp = []
    if kv_layers:
        saved_attn = patch_attention_layers(model, kv_layers, kv_bits, kv_rotation, kv_alpha, seed)
    if mlp_params:
        saved_mlp = patch_mlp_layers(model, mlp_params, mlp_bits, mlp_block_size, seed)
    metrics = summarize_candidate(model, eval_batch, baseline_logits, baseline, topk)
    restore_forwards(saved_mlp)
    restore_forwards(saved_attn)
    return metrics


def run_probe(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.float32)
    model.eval()

    calibration_prompts = load_prompts(args.calibration_prompts, CALIBRATION_PROMPTS)
    eval_prompts = load_prompts(args.eval_prompts, EVAL_PROMPTS)
    cal_batch = tokenize(tokenizer, calibration_prompts, args.max_length)
    eval_batch = tokenize(tokenizer, eval_prompts, args.max_length)
    baseline = evaluate_model(model, eval_batch, topk=args.topk)
    baseline_logits = baseline.pop("logits")

    kv_layers = parse_int_list(args.kv_layers)
    mlp_layers = parse_int_list(args.mlp_layers)
    args.bits = args.mlp_bits
    calibrated_mlp = calibrate_layer_params(args, model, cal_batch, mlp_layers)
    identity_mlp = [{"layer": layer, "rotation": "identity", "alpha": 1.0} for layer in mlp_layers]
    clipped_mlp = [{"layer": layer, "rotation": "identity", "alpha": args.identity_alpha} for layer in mlp_layers]

    candidates = []
    specs = [
        {
            "candidate": f"kv_identity_{args.kv_bits}bit",
            "kv_rotation": "identity",
            "mlp_params": [],
        },
        {
            "candidate": f"kv_{args.kv_rotation}_{args.kv_bits}bit",
            "kv_rotation": args.kv_rotation,
            "mlp_params": [],
        },
        {
            "candidate": "mlp_identity_absmax_2bit",
            "kv_rotation": None,
            "mlp_params": identity_mlp,
        },
        {
            "candidate": f"mlp_identity_alpha_{args.identity_alpha:g}_2bit",
            "kv_rotation": None,
            "mlp_params": clipped_mlp,
        },
        {
            "candidate": "mlp_calibrated_rotation_2bit",
            "kv_rotation": None,
            "mlp_params": calibrated_mlp,
        },
        {
            "candidate": f"fused_kv_{args.kv_rotation}_{args.kv_bits}bit_mlp_calibrated_2bit",
            "kv_rotation": args.kv_rotation,
            "mlp_params": calibrated_mlp,
        },
        {
            "candidate": f"fused_kv_identity_{args.kv_bits}bit_mlp_identity_absmax_2bit",
            "kv_rotation": "identity",
            "mlp_params": identity_mlp,
        },
    ]

    for spec in specs:
        use_kv = spec["kv_rotation"] is not None
        metrics = run_candidate(
            model=model,
            eval_batch=eval_batch,
            baseline_logits=baseline_logits,
            baseline=baseline,
            topk=args.topk,
            kv_layers=kv_layers if use_kv else [],
            kv_bits=args.kv_bits,
            kv_rotation=spec["kv_rotation"] or "identity",
            kv_alpha=args.kv_alpha,
            mlp_params=spec["mlp_params"],
            mlp_bits=args.mlp_bits,
            mlp_block_size=args.block_size,
            seed=args.seed,
        )
        candidates.append({"candidate": spec["candidate"], "metrics": metrics})
        print(
            f"eval {spec['candidate']} loss={metrics['loss']:.6f} "
            f"kl={metrics['kl_from_baseline']:.6f} topk={metrics['topk_overlap']:.6f}"
        )

    return {
        "experiment": "fused_stack_probe",
        "repo": args.repo,
        "kv_layers": kv_layers,
        "kv_bits": args.kv_bits,
        "kv_rotation": args.kv_rotation,
        "mlp_layers": mlp_layers,
        "mlp_bits": args.mlp_bits,
        "block_size": args.block_size,
        "calibration_prompt_count": len(calibration_prompts),
        "eval_prompt_count": len(eval_prompts),
        "baseline_eval": baseline,
        "calibrated_mlp_params": calibrated_mlp,
        "candidates": candidates,
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_fused_stack_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end fused KV + MLP activation quantization stack probe")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--kv-layers", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29")
    parser.add_argument("--kv-bits", type=int, default=3)
    parser.add_argument("--kv-rotation", default="hadamard")
    parser.add_argument("--kv-alpha", type=float, default=1.0)
    parser.add_argument("--mlp-layers", default="16,17,12,6,14,21,5,19")
    parser.add_argument("--mlp-bits", type=int, default=2)
    parser.add_argument("--rotations", default="identity,block_hadamard,block_hadamard_sign_perm")
    parser.add_argument("--alphas", default="1.0,0.75,0.5,0.375,0.25")
    parser.add_argument("--identity-alpha", type=float, default=0.375)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--calibration-prompts")
    parser.add_argument("--eval-prompts")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    result = run_probe(args)
    path = save_result(result, args.out_dir)
    print(json.dumps({"baseline_eval": result["baseline_eval"], "candidates": result["candidates"]}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
