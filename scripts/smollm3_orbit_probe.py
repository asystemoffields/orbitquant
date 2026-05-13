import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open


DEFAULT_REPO = "HuggingFaceTB/SmolLM3-3B"


@dataclass
class ModelShape:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def query_heads_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


def download_json(repo: str, filename: str) -> dict:
    path = hf_hub_download(repo, filename)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_shape(repo: str) -> ModelShape:
    config = download_json(repo, "config.json")
    return ModelShape(
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
    )


def tensor_path(repo: str, index: dict, tensor_name: str) -> str:
    shard = index["weight_map"][tensor_name]
    return hf_hub_download(repo, shard)


def load_tensor(repo: str, index: dict, tensor_name: str) -> torch.Tensor:
    path = tensor_path(repo, index, tensor_name)
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_name).to(torch.float32)


def quantize_symmetric_per_row_group(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("Expected a 2D tensor")
    if bits < 2:
        raise ValueError("Use at least 2 bits for this symmetric quantizer")

    rows, cols = weight.shape
    qmax = (1 << (bits - 1)) - 1
    pad = (group_size - (cols % group_size)) % group_size
    if pad:
        padded = F.pad(weight, (0, pad))
    else:
        padded = weight

    grouped = padded.reshape(rows, -1, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(grouped / scale).clamp(-qmax, qmax)
    dequant = (q * scale).reshape(rows, cols + pad)
    return dequant[:, :cols].contiguous()


def rel_mse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    numerator = torch.mean((reference - candidate) ** 2)
    denominator = torch.mean(reference**2).clamp_min(1e-12)
    return float((numerator / denominator).item())


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def hadamard(dim: int) -> torch.Tensor:
    if not is_power_of_two(dim):
        raise ValueError("Hadamard candidate requires a power-of-two dimension")
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
    signs = torch.sign(torch.diag(r)).clamp(min=-1, max=1)
    signs[signs == 0] = 1
    return q * signs


def rotation_candidates(dim: int, trials: int, seed: int) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    candidates: list[tuple[str, torch.Tensor]] = [("identity", torch.eye(dim))]

    candidates.append(("sign_perm", random_sign_perm(dim, generator)))

    if is_power_of_two(dim):
        h = hadamard(dim)
        candidates.append(("hadamard", h))
        candidates.append(("hadamard_sign_perm", h @ random_sign_perm(dim, generator)))

    for idx in range(trials):
        candidates.append((f"random_orthogonal_{idx}", random_orthogonal(dim, generator)))

    return candidates


def transform_value_o(
    value_weight: torch.Tensor,
    o_segments: list[torch.Tensor],
    rotation: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    transformed_value = rotation.T @ value_weight
    transformed_o = [segment @ rotation for segment in o_segments]
    return transformed_value.contiguous(), [segment.contiguous() for segment in transformed_o]


def value_o_output(
    x: torch.Tensor,
    value_weight: torch.Tensor,
    o_segments: list[torch.Tensor],
) -> torch.Tensor:
    value = x @ value_weight.T
    output = torch.zeros(x.shape[0], o_segments[0].shape[0])
    for segment in o_segments:
        output += value @ segment.T
    return output


def run_value_o_probe(args, repo: str, index: dict, shape: ModelShape) -> dict:
    head_dim = shape.head_dim
    if args.kv_head < 0 or args.kv_head >= shape.num_key_value_heads:
        raise ValueError(f"kv_head must be in [0, {shape.num_key_value_heads})")

    layer = args.layer
    v_name = f"model.layers.{layer}.self_attn.v_proj.weight"
    o_name = f"model.layers.{layer}.self_attn.o_proj.weight"
    v_proj = load_tensor(repo, index, v_name)
    o_proj = load_tensor(repo, index, o_name)

    kv_start = args.kv_head * head_dim
    kv_stop = kv_start + head_dim
    value_head = v_proj[kv_start:kv_stop, :]

    group_start = args.kv_head * shape.query_heads_per_kv
    o_segments = []
    for query_head in range(group_start, group_start + shape.query_heads_per_kv):
        col_start = query_head * head_dim
        col_stop = col_start + head_dim
        o_segments.append(o_proj[:, col_start:col_stop])

    generator = torch.Generator().manual_seed(args.seed)
    x = torch.randn(args.samples, shape.hidden_size, generator=generator) / math.sqrt(shape.hidden_size)
    reference = value_o_output(x, value_head, o_segments)

    rows = []
    for name, rotation in rotation_candidates(head_dim, args.trials, args.seed + 17):
        v_t, o_t = transform_value_o(value_head, o_segments, rotation)
        exact = value_o_output(x, v_t, o_t)

        q_v = quantize_symmetric_per_row_group(v_t, args.bits, args.group_size)
        q_o = [
            quantize_symmetric_per_row_group(segment, args.bits, args.group_size)
            for segment in o_t
        ]
        quantized = value_o_output(x, q_v, q_o)

        rows.append(
            {
                "candidate": name,
                "exact_rel_mse": rel_mse(reference, exact),
                "quantized_rel_mse": rel_mse(reference, quantized),
            }
        )

    rows.sort(key=lambda row: row["quantized_rel_mse"])
    return {
        "experiment": "value_o",
        "layer": layer,
        "kv_head": args.kv_head,
        "bits": args.bits,
        "group_size": args.group_size,
        "samples": args.samples,
        "results": rows,
    }


def channel_scalings(
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    trials: int,
    seed: int,
) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    up_norm = up_weight.pow(2).mean(dim=1).sqrt().clamp_min(1e-8)
    down_norm = down_weight.pow(2).mean(dim=0).sqrt().clamp_min(1e-8)
    up_max = up_weight.abs().amax(dim=1).clamp_min(1e-8)
    down_max = down_weight.abs().amax(dim=0).clamp_min(1e-8)

    def clipped(values: torch.Tensor) -> torch.Tensor:
        values = values / torch.exp(torch.mean(torch.log(values.clamp_min(1e-8))))
        return values.clamp(0.25, 4.0)

    candidates = [
        ("identity", torch.ones(up_weight.shape[0])),
        ("norm_balance", clipped(torch.sqrt(down_norm / up_norm))),
        ("max_balance", clipped(torch.sqrt(down_max / up_max))),
    ]

    for idx in range(trials):
        noise = torch.randn(up_weight.shape[0], generator=generator) * 0.35
        candidates.append((f"lognormal_{idx}", clipped(torch.exp(noise))))

    return candidates


def mlp_output(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    gate = F.silu(x @ gate_weight.T)
    up = x @ up_weight.T
    return (gate * up) @ down_weight.T


def run_mlp_scale_probe(args, repo: str, index: dict, shape: ModelShape) -> dict:
    layer = args.layer
    gate = load_tensor(repo, index, f"model.layers.{layer}.mlp.gate_proj.weight")
    up = load_tensor(repo, index, f"model.layers.{layer}.mlp.up_proj.weight")
    down = load_tensor(repo, index, f"model.layers.{layer}.mlp.down_proj.weight")

    generator = torch.Generator().manual_seed(args.seed + 101)
    sample_count = min(args.samples, 48)
    x = torch.randn(sample_count, shape.hidden_size, generator=generator) / math.sqrt(shape.hidden_size)
    reference = mlp_output(x, gate, up, down)

    rows = []
    for name, scales in channel_scalings(up, down, args.trials, args.seed + 211):
        up_t = (up * scales[:, None]).contiguous()
        down_t = (down / scales[None, :]).contiguous()
        exact = mlp_output(x, gate, up_t, down_t)

        q_up = quantize_symmetric_per_row_group(up_t, args.bits, args.group_size)
        q_down = quantize_symmetric_per_row_group(down_t, args.bits, args.group_size)
        quantized = mlp_output(x, gate, q_up, q_down)

        rows.append(
            {
                "candidate": name,
                "exact_rel_mse": rel_mse(reference, exact),
                "quantized_rel_mse": rel_mse(reference, quantized),
            }
        )

    rows.sort(key=lambda row: row["quantized_rel_mse"])
    return {
        "experiment": "mlp_scale",
        "layer": layer,
        "bits": args.bits,
        "group_size": args.group_size,
        "samples": sample_count,
        "results": rows,
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{payload['experiment']}_layer{payload['layer']}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolLM3 orbit-aware quantization probes")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--experiment", choices=["value_o", "mlp_scale", "both"], default="value_o")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--kv-head", type=int, default=0)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    repo = args.repo
    index = download_json(repo, "model.safetensors.index.json")
    shape = load_shape(repo)

    all_results = []
    if args.experiment in {"value_o", "both"}:
        all_results.append(run_value_o_probe(args, repo, index, shape))
    if args.experiment in {"mlp_scale", "both"}:
        all_results.append(run_mlp_scale_probe(args, repo, index, shape))

    for result in all_results:
        path = save_result(result, args.out_dir)
        print(json.dumps(result, indent=2))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
