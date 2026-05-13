import argparse
import json
import time
import types
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import DEFAULT_REPO, evaluate_model, get_layer, tokenize
from smol135_mlp_activation_rotation_probe import capture_mlp_intermediate
from smol135_mlp_hadamard_plus_probe import load_prompts, make_rotation
from smol135_mlp_stack_probe import restore_forwards, summarize_candidate
from smol135_sweep import EVAL_PROMPTS, parse_int_list


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quantize_with_rotation(x: torch.Tensor, bits: int, rotation: torch.Tensor, alpha: float) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    rotation = rotation.to(device=x.device, dtype=x.dtype)
    rotated = x @ rotation
    scale = rotated.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) * alpha / qmax
    quantized = torch.round(rotated / scale).clamp(-qmax, qmax) * scale
    return quantized @ rotation.T


def make_mlp_forward(mlp, bits: int, rotation: torch.Tensor, alpha: float):
    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        z = gate * up
        z_quant = quantize_with_rotation(z, bits, rotation, alpha)
        return self.down_proj(z_quant)

    return types.MethodType(forward, mlp)


def patch_mlp_precomputed(model, params: list[dict], bits: int) -> list:
    saved = []
    for item in params:
        layer = get_layer(model, item["layer"])
        original_forward = layer.mlp.forward
        layer.mlp.forward = make_mlp_forward(layer.mlp, bits, item["rotation_tensor"], item["alpha"])
        saved.append((layer.mlp, original_forward))
    return saved


def serializable_param(item: dict) -> dict:
    return {
        "layer": item["layer"],
        "rotation": item["rotation"],
        "alpha": item["alpha"],
        "local_calibration_down_output_rel_mse": item.get("local_calibration_down_output_rel_mse"),
        "local_eval_down_output_rel_mse": item.get("local_eval_down_output_rel_mse"),
    }


def local_param(row: dict, mode: str) -> dict:
    if mode == "best":
        item = row["summary"]["best_by_calibration"]
    elif mode == "plain":
        item = row["summary"]["plain_block_hadamard_best"]
    elif mode == "identity":
        item = row["summary"]["identity_absmax"]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return {
        "layer": row["layer"],
        "rotation": item["rotation"],
        "alpha": item["alpha"],
        "local_calibration_down_output_rel_mse": item["calibration"]["down_output_rel_mse"],
        "local_eval_down_output_rel_mse": item["eval"]["down_output_rel_mse"],
    }


def build_layer_params(args, model, cal_batch, local_rows: list[dict], mode: str) -> list[dict]:
    params = []
    for row in local_rows:
        layer_idx = row["layer"]
        item = local_param(row, mode)
        layer = get_layer(model, layer_idx)
        down_weight = layer.mlp.down_proj.weight.detach().float().cpu()
        cal_z, _ = capture_mlp_intermediate(model, cal_batch, layer_idx)
        item["rotation_tensor"] = make_rotation(
            item["rotation"],
            cal_z,
            down_weight,
            args.block_size,
            args.seed + layer_idx * 1009 + len(item["rotation"]),
        )
        params.append(item)
        print(
            f"build {mode} layer={layer_idx} rotation={item['rotation']} "
            f"alpha={item['alpha']:.3f} local_eval_down={item['local_eval_down_output_rel_mse']:.6f}"
        )
    return params


def run_model_candidate(model, eval_batch, baseline_logits, baseline, topk, params, bits) -> dict:
    saved = patch_mlp_precomputed(model, params, bits)
    metrics = summarize_candidate(model, eval_batch, baseline_logits, baseline, topk)
    restore_forwards(saved)
    return metrics


def run_probe(args) -> dict:
    local_result = load_json(args.local_result)
    local_rows = sorted(local_result["rows"], key=lambda row: row["layer"])
    if args.layers:
        wanted = set(parse_int_list(args.layers))
        local_rows = [row for row in local_rows if row["layer"] in wanted]

    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.float32)
    model.eval()

    calibration_prompts = load_prompts(args.calibration_prompts, [])
    eval_prompts = load_prompts(args.eval_prompts, EVAL_PROMPTS)
    cal_batch = tokenize(tokenizer, calibration_prompts, args.max_length)
    eval_batch = tokenize(tokenizer, eval_prompts, args.max_length)
    baseline = evaluate_model(model, eval_batch, topk=args.topk)
    baseline_logits = baseline.pop("logits")

    best_params = build_layer_params(args, model, cal_batch, local_rows, "best")
    plain_params = build_layer_params(args, model, cal_batch, local_rows, "plain")
    identity_params = build_layer_params(args, model, cal_batch, local_rows, "identity")

    rows = []
    for best, plain, identity in zip(best_params, plain_params, identity_params):
        identity_metrics = run_model_candidate(model, eval_batch, baseline_logits, baseline, args.topk, [identity], args.bits)
        plain_metrics = run_model_candidate(model, eval_batch, baseline_logits, baseline, args.topk, [plain], args.bits)
        best_metrics = run_model_candidate(model, eval_batch, baseline_logits, baseline, args.topk, [best], args.bits)
        row = {
            "layer": best["layer"],
            "identity": {"param": serializable_param(identity), "metrics": identity_metrics},
            "plain_block_hadamard": {"param": serializable_param(plain), "metrics": plain_metrics},
            "hadamard_plus": {"param": serializable_param(best), "metrics": best_metrics},
            "hadamard_plus_minus_plain": {
                "kl": best_metrics["kl_from_baseline"] - plain_metrics["kl_from_baseline"],
                "topk_overlap": best_metrics["topk_overlap"] - plain_metrics["topk_overlap"],
                "loss_delta_from_baseline": best_metrics["loss_delta_from_baseline"]
                - plain_metrics["loss_delta_from_baseline"],
            },
        }
        rows.append(row)
        print(
            f"impact layer={best['layer']} plain_kl={plain_metrics['kl_from_baseline']:.6f} "
            f"plus_kl={best_metrics['kl_from_baseline']:.6f} "
            f"delta={row['hadamard_plus_minus_plain']['kl']:.6f}"
        )

    sorted_best = sorted(rows, key=lambda row: row["hadamard_plus"]["metrics"]["kl_from_baseline"])
    sorted_plain = sorted(rows, key=lambda row: row["plain_block_hadamard"]["metrics"]["kl_from_baseline"])
    stacks = []
    for count in parse_int_list(args.top_counts):
        best_layers = [row["layer"] for row in sorted_best[:count]]
        plain_layers = [row["layer"] for row in sorted_plain[:count]]
        best_stack = [item for item in best_params if item["layer"] in best_layers]
        plain_stack = [item for item in plain_params if item["layer"] in plain_layers]
        same_layer_plain_stack = [item for item in plain_params if item["layer"] in best_layers]

        best_metrics = run_model_candidate(model, eval_batch, baseline_logits, baseline, args.topk, best_stack, args.bits)
        plain_metrics = run_model_candidate(model, eval_batch, baseline_logits, baseline, args.topk, plain_stack, args.bits)
        same_layer_plain_metrics = run_model_candidate(
            model, eval_batch, baseline_logits, baseline, args.topk, same_layer_plain_stack, args.bits
        )
        stacks.append(
            {
                "top_count": count,
                "hadamard_plus_layers": best_layers,
                "plain_layers": plain_layers,
                "hadamard_plus_metrics": best_metrics,
                "plain_block_hadamard_metrics": plain_metrics,
                "same_layer_plain_block_hadamard_metrics": same_layer_plain_metrics,
                "plus_minus_plain_top_layers": {
                    "kl": best_metrics["kl_from_baseline"] - plain_metrics["kl_from_baseline"],
                    "topk_overlap": best_metrics["topk_overlap"] - plain_metrics["topk_overlap"],
                    "loss_delta_from_baseline": best_metrics["loss_delta_from_baseline"]
                    - plain_metrics["loss_delta_from_baseline"],
                },
                "plus_minus_same_layers_plain": {
                    "kl": best_metrics["kl_from_baseline"] - same_layer_plain_metrics["kl_from_baseline"],
                    "topk_overlap": best_metrics["topk_overlap"] - same_layer_plain_metrics["topk_overlap"],
                    "loss_delta_from_baseline": best_metrics["loss_delta_from_baseline"]
                    - same_layer_plain_metrics["loss_delta_from_baseline"],
                },
            }
        )
        print(
            f"stack top_count={count} plus_layers={best_layers} "
            f"plus_kl={best_metrics['kl_from_baseline']:.6f} "
            f"plain_kl={plain_metrics['kl_from_baseline']:.6f}"
        )

    wins_vs_plain = sum(1 for row in rows if row["hadamard_plus_minus_plain"]["kl"] < 0)
    return {
        "experiment": "mlp_hadamard_plus_impact_sweep",
        "repo": args.repo,
        "local_result": str(args.local_result),
        "bits": args.bits,
        "block_size": args.block_size,
        "calibration_prompt_count": len(calibration_prompts),
        "eval_prompt_count": len(eval_prompts),
        "baseline_eval": baseline,
        "summary": {
            "groups": len(rows),
            "single_layer_kl_wins_vs_plain": wins_vs_plain,
            "mean_single_layer_kl_delta_vs_plain": sum(row["hadamard_plus_minus_plain"]["kl"] for row in rows)
            / len(rows),
            "mean_single_layer_topk_delta_vs_plain": sum(
                row["hadamard_plus_minus_plain"]["topk_overlap"] for row in rows
            )
            / len(rows),
        },
        "rows": rows,
        "stacks": stacks,
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_mlp_hadamard_plus_impact_{payload['bits']}bit_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end impact sweep for Hadamard-plus MLP bus candidates")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--local-result", type=Path, required=True)
    parser.add_argument("--layers")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--top-counts", default="4,8,12")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--calibration-prompts", required=True)
    parser.add_argument("--eval-prompts")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    result = run_probe(args)
    path = save_result(result, args.out_dir)
    print(json.dumps({"experiment": result["experiment"], "summary": result["summary"], "stacks": result["stacks"]}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
