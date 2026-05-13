import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import DEFAULT_REPO, get_layer, hadamard, tokenize
from smol135_sweep import EVAL_PROMPTS, parse_int_list, parse_str_list


def block_diag_hadamard(dim: int, block_size: int) -> torch.Tensor:
    if dim % block_size:
        raise ValueError("dim must be divisible by block_size")
    h = hadamard(block_size)
    blocks = [h for _ in range(dim // block_size)]
    return torch.block_diag(*blocks)


def random_sign_perm(dim: int, generator: torch.Generator) -> torch.Tensor:
    perm = torch.randperm(dim, generator=generator)
    signs = torch.randint(0, 2, (dim,), generator=generator, dtype=torch.float32)
    signs = signs.mul(2).sub(1)
    matrix = torch.zeros(dim, dim)
    matrix[torch.arange(dim), perm] = signs
    return matrix


def rotation_matrix(dim: int, rotation: str, block_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if rotation == "identity":
        return torch.eye(dim)
    if rotation == "block_hadamard":
        return block_diag_hadamard(dim, block_size)
    if rotation == "block_hadamard_sign_perm":
        return block_diag_hadamard(dim, block_size) @ random_sign_perm(dim, generator)
    if rotation == "sign_perm":
        return random_sign_perm(dim, generator)
    raise ValueError(f"Unknown rotation: {rotation}")


def quantize_vectors(vectors: torch.Tensor, bits: int, rotation: torch.Tensor) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    rotated = vectors @ rotation
    scale = rotated.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    quantized = torch.round(rotated / scale).clamp(-qmax, qmax) * scale
    return quantized @ rotation.T


def rel_mse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((((reference - candidate) ** 2).mean() / (reference**2).mean().clamp_min(1e-12)).item())


def capture_mlp_intermediate(model, batch, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_layer(model, layer_idx)
    captured = {}

    def hook(_module, inputs, _output):
        x = inputs[0].detach().float()
        gate = F.silu(layer.mlp.gate_proj(x))
        up = layer.mlp.up_proj(x)
        z = gate * up
        y = layer.mlp.down_proj(z)
        captured["z"] = z.reshape(-1, z.shape[-1]).detach().float().cpu()
        captured["y"] = y.reshape(-1, y.shape[-1]).detach().float().cpu()

    handle = layer.mlp.register_forward_hook(hook)
    with torch.no_grad():
        model(**batch)
    handle.remove()
    return captured["z"], captured["y"]


def run_probe(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.float32)
    model.eval()
    batch = tokenize(tokenizer, EVAL_PROMPTS, args.max_length)
    layers = parse_int_list(args.layers)
    rotations = parse_str_list(args.rotations)
    rows = []

    for layer_idx in layers:
        layer = get_layer(model, layer_idx)
        z, reference_y = capture_mlp_intermediate(model, batch, layer_idx)
        down_weight = layer.mlp.down_proj.weight.detach().float().cpu()

        for rotation_name in rotations:
            rot = rotation_matrix(
                z.shape[-1],
                rotation_name,
                args.block_size,
                args.seed + layer_idx * 1009 + len(rotation_name),
            )
            z_quant = quantize_vectors(z, args.bits, rot)
            candidate_y = z_quant @ down_weight.T
            rows.append(
                {
                    "layer": layer_idx,
                    "rotation": rotation_name,
                    "bits": args.bits,
                    "activation_rel_mse": rel_mse(z, z_quant),
                    "down_output_rel_mse": rel_mse(reference_y, candidate_y),
                }
            )

        group_rows = [row for row in rows if row["layer"] == layer_idx]
        best = min(group_rows, key=lambda row: row["down_output_rel_mse"])
        print(
            f"mlp layer={layer_idx} best={best['rotation']} "
            f"down_mse={best['down_output_rel_mse']:.6f}"
        )

    grouped = {}
    for row in rows:
        grouped.setdefault(row["layer"], []).append(row)

    wins = {rotation: 0 for rotation in rotations}
    activation_deltas = []
    down_deltas = []
    for group_rows in grouped.values():
        identity = next(row for row in group_rows if row["rotation"] == "identity")
        best = min(group_rows, key=lambda row: row["down_output_rel_mse"])
        wins[best["rotation"]] += 1
        activation_deltas.append(best["activation_rel_mse"] - identity["activation_rel_mse"])
        down_deltas.append(best["down_output_rel_mse"] - identity["down_output_rel_mse"])

    return {
        "experiment": "mlp_activation_rotation_probe",
        "repo": args.repo,
        "layers": layers,
        "bits": args.bits,
        "rotations": rotations,
        "block_size": args.block_size,
        "eval_prompt_count": len(EVAL_PROMPTS),
        "summary": {
            "groups": len(grouped),
            "best_rotation_counts": wins,
            "mean_best_activation_delta_vs_identity": sum(activation_deltas) / len(activation_deltas),
            "mean_best_down_output_delta_vs_identity": sum(down_deltas) / len(down_deltas),
        },
        "rows": rows,
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_mlp_activation_rotation_{payload['bits']}bit_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant-style rotation probe on Smol135 MLP activations")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--layers", default="0,1,2,4,8,12,16,20,24,29")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--rotations", default="identity,block_hadamard,block_hadamard_sign_perm")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    result = run_probe(args)
    path = save_result(result, args.out_dir)
    print(json.dumps({"experiment": result["experiment"], "summary": result["summary"]}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
