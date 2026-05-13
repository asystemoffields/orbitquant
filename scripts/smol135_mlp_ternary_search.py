import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import DEFAULT_REPO, get_layer, tokenize
from smol135_mlp_activation_rotation_probe import capture_mlp_intermediate, rel_mse, rotation_matrix
from smol135_sweep import CALIBRATION_PROMPTS, EVAL_PROMPTS, parse_float_list, parse_int_list, parse_str_list


def quantize_vectors_alpha(
    vectors: torch.Tensor,
    bits: int,
    rotation: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    rotated = vectors @ rotation
    scale = rotated.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) * alpha / qmax
    quantized = torch.round(rotated / scale).clamp(-qmax, qmax) * scale
    return quantized @ rotation.T


def evaluate_candidate(z, y, down_weight, bits, rotation, alpha) -> dict:
    z_quant = quantize_vectors_alpha(z, bits, rotation, alpha)
    y_quant = z_quant @ down_weight.T
    return {
        "activation_rel_mse": rel_mse(z, z_quant),
        "down_output_rel_mse": rel_mse(y, y_quant),
    }


def run_search(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.float32)
    model.eval()
    cal_batch = tokenize(tokenizer, CALIBRATION_PROMPTS, args.max_length)
    eval_batch = tokenize(tokenizer, EVAL_PROMPTS, args.max_length)

    layers = parse_int_list(args.layers)
    rotations = parse_str_list(args.rotations)
    alphas = parse_float_list(args.alphas)
    rows = []

    for layer_idx in layers:
        layer = get_layer(model, layer_idx)
        down_weight = layer.mlp.down_proj.weight.detach().float().cpu()
        cal_z, cal_y = capture_mlp_intermediate(model, cal_batch, layer_idx)
        eval_z, eval_y = capture_mlp_intermediate(model, eval_batch, layer_idx)

        candidates = []
        for rotation_name in rotations:
            rot = rotation_matrix(
                cal_z.shape[-1],
                rotation_name,
                args.block_size,
                args.seed + layer_idx * 1009 + len(rotation_name),
            )
            for alpha in alphas:
                cal = evaluate_candidate(cal_z, cal_y, down_weight, args.bits, rot, alpha)
                ev = evaluate_candidate(eval_z, eval_y, down_weight, args.bits, rot, alpha)
                candidates.append(
                    {
                        "rotation": rotation_name,
                        "alpha": alpha,
                        "calibration": cal,
                        "eval": ev,
                    }
                )

        identity = next(
            item for item in candidates if item["rotation"] == "identity" and abs(item["alpha"] - 1.0) < 1e-9
        )
        best = min(candidates, key=lambda item: item["calibration"]["down_output_rel_mse"])
        rows.append(
            {
                "layer": layer_idx,
                "identity": identity,
                "best_by_calibration": best,
                "best_minus_identity_eval": {
                    "activation_rel_mse": best["eval"]["activation_rel_mse"]
                    - identity["eval"]["activation_rel_mse"],
                    "down_output_rel_mse": best["eval"]["down_output_rel_mse"]
                    - identity["eval"]["down_output_rel_mse"],
                },
            }
        )
        print(
            f"mlp2 layer={layer_idx} "
            f"identity_eval={identity['eval']['down_output_rel_mse']:.6f} "
            f"best={best['rotation']} alpha={best['alpha']:.3f} "
            f"best_eval={best['eval']['down_output_rel_mse']:.6f}"
        )

    wins = sum(1 for row in rows if row["best_minus_identity_eval"]["down_output_rel_mse"] < 0)
    activation_wins = sum(1 for row in rows if row["best_minus_identity_eval"]["activation_rel_mse"] < 0)
    best_counts = {}
    for row in rows:
        key = f"{row['best_by_calibration']['rotation']}@{row['best_by_calibration']['alpha']}"
        best_counts[key] = best_counts.get(key, 0) + 1

    return {
        "experiment": "mlp_ternary_scale_search",
        "repo": args.repo,
        "layers": layers,
        "bits": args.bits,
        "rotations": rotations,
        "alphas": alphas,
        "block_size": args.block_size,
        "summary": {
            "groups": len(rows),
            "down_output_wins": wins,
            "activation_wins": activation_wins,
            "best_counts": best_counts,
            "mean_eval_activation_delta_vs_identity": sum(
                row["best_minus_identity_eval"]["activation_rel_mse"] for row in rows
            )
            / len(rows),
            "mean_eval_down_output_delta_vs_identity": sum(
                row["best_minus_identity_eval"]["down_output_rel_mse"] for row in rows
            )
            / len(rows),
        },
        "rows": rows,
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_mlp_ternary_search_{payload['bits']}bit_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration-selected MLP 2-bit scale/rotation search")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--layers", default="0,1,2,4,8,12,16,20,24,29")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--rotations", default="identity,block_hadamard,block_hadamard_sign_perm")
    parser.add_argument("--alphas", default="1.0,0.75,0.5,0.375,0.25")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    result = run_search(args)
    path = save_result(result, args.out_dir)
    print(json.dumps({"experiment": result["experiment"], "summary": result["summary"]}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
