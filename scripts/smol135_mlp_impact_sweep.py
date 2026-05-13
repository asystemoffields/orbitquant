import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import DEFAULT_REPO, evaluate_model, tokenize
from smol135_mlp_stack_probe import calibrate_layer_params, patch_mlp_layers, restore_forwards, summarize_candidate
from smol135_sweep import CALIBRATION_PROMPTS, EVAL_PROMPTS, parse_float_list, parse_int_list


def load_prompts(path: str | None, fallback: list[str]) -> list[str]:
    if path is None:
        return fallback
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def run_sweep(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.float32)
    model.eval()

    layers = parse_int_list(args.layers)
    calibration_prompts = load_prompts(args.calibration_prompts, CALIBRATION_PROMPTS)
    eval_prompts = load_prompts(args.eval_prompts, EVAL_PROMPTS)
    cal_batch = tokenize(tokenizer, calibration_prompts, args.max_length)
    eval_batch = tokenize(tokenizer, eval_prompts, args.max_length)
    baseline = evaluate_model(model, eval_batch, topk=args.topk)
    baseline_logits = baseline.pop("logits")

    calibrated_params = calibrate_layer_params(args, model, cal_batch, layers)
    by_layer = {item["layer"]: item for item in calibrated_params}

    rows = []
    for layer_idx in layers:
        identity_param = [{"layer": layer_idx, "rotation": "identity", "alpha": 1.0}]
        calibrated_param = [by_layer[layer_idx]]

        saved = patch_mlp_layers(model, identity_param, args.bits, args.block_size, args.seed)
        identity = summarize_candidate(model, eval_batch, baseline_logits, baseline, args.topk)
        restore_forwards(saved)

        saved = patch_mlp_layers(model, calibrated_param, args.bits, args.block_size, args.seed)
        calibrated = summarize_candidate(model, eval_batch, baseline_logits, baseline, args.topk)
        restore_forwards(saved)

        rows.append(
            {
                "layer": layer_idx,
                "calibrated_param": by_layer[layer_idx],
                "identity": identity,
                "calibrated": calibrated,
                "calibrated_minus_identity": {
                    "kl": calibrated["kl_from_baseline"] - identity["kl_from_baseline"],
                    "topk_overlap": calibrated["topk_overlap"] - identity["topk_overlap"],
                    "loss_delta_abs": abs(calibrated["loss_delta_from_baseline"])
                    - abs(identity["loss_delta_from_baseline"]),
                },
            }
        )
        print(
            f"impact layer={layer_idx} identity_kl={identity['kl_from_baseline']:.6f} "
            f"cal_kl={calibrated['kl_from_baseline']:.6f}"
        )

    thresholds = parse_float_list(args.thresholds)
    stacks = []
    for threshold in thresholds:
        selected = [
            row["layer"]
            for row in rows
            if row["calibrated"]["kl_from_baseline"] <= threshold
        ]
        params = [by_layer[layer] for layer in selected]
        saved = patch_mlp_layers(model, params, args.bits, args.block_size, args.seed)
        metrics = summarize_candidate(model, eval_batch, baseline_logits, baseline, args.topk)
        restore_forwards(saved)
        stacks.append({"threshold": threshold, "layers": selected, "metrics": metrics})
        print(
            f"stack threshold={threshold:.4f} layers={len(selected)} "
            f"kl={metrics['kl_from_baseline']:.6f} topk={metrics['topk_overlap']:.6f}"
        )

    sorted_layers = sorted(rows, key=lambda row: row["calibrated"]["kl_from_baseline"])
    for count in parse_int_list(args.top_counts):
        selected = [row["layer"] for row in sorted_layers[:count]]
        params = [by_layer[layer] for layer in selected]
        saved = patch_mlp_layers(model, params, args.bits, args.block_size, args.seed)
        metrics = summarize_candidate(model, eval_batch, baseline_logits, baseline, args.topk)
        restore_forwards(saved)
        stacks.append({"top_count": count, "layers": selected, "metrics": metrics})
        print(
            f"stack top_count={count} layers={selected} "
            f"kl={metrics['kl_from_baseline']:.6f} topk={metrics['topk_overlap']:.6f}"
        )

    return {
        "experiment": "mlp_impact_sweep",
        "repo": args.repo,
        "layers": layers,
        "bits": args.bits,
        "calibration_prompt_count": len(calibration_prompts),
        "eval_prompt_count": len(eval_prompts),
        "baseline_eval": baseline,
        "rows": rows,
        "stacks": stacks,
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_mlp_impact_{payload['bits']}bit_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-layer end-to-end MLP quantization impact sweep")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--layers", default="0,1,2,4,8,12,16,20,24,29")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--rotations", default="identity,block_hadamard,block_hadamard_sign_perm")
    parser.add_argument("--alphas", default="1.0,0.75,0.5,0.375,0.25")
    parser.add_argument("--thresholds", default="0.005,0.01,0.02,0.05,0.10")
    parser.add_argument("--top-counts", default="4,8,12,16")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--calibration-prompts")
    parser.add_argument("--eval-prompts")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    result = run_sweep(args)
    path = save_result(result, args.out_dir)
    summary = {
        "baseline_eval": result["baseline_eval"],
        "best_layers": [
            {
                "layer": row["layer"],
                "cal_kl": row["calibrated"]["kl_from_baseline"],
                "cal_topk": row["calibrated"]["topk_overlap"],
            }
            for row in sorted(result["rows"], key=lambda item: item["calibrated"]["kl_from_baseline"])[:10]
        ],
        "stacks": result["stacks"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
