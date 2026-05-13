import argparse
import itertools
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import DEFAULT_REPO, evaluate_model, tokenize
from smol135_fused_choice_allocator import run_stack
from smol135_mlp_hadamard_plus_impact_sweep import build_layer_params, load_json
from smol135_mlp_hadamard_plus_probe import load_prompts
from smol135_sweep import EVAL_PROMPTS, parse_int_list


def make_kv_specs(layers: list[int], rotation: str) -> list[dict]:
    return [{"layer": layer, "rotation": rotation} for layer in layers]


def serializable_params(params: list[dict]) -> list[dict]:
    return [
        {
            "layer": item["layer"],
            "primitive": item["primitive"],
            "rotation": item["rotation"],
            "alpha": item["alpha"],
        }
        for item in params
    ]


def run_probe(args) -> dict:
    local_payload = load_json(args.local_result)
    mlp_layers = parse_int_list(args.mlp_layers)
    kv_layers = parse_int_list(args.kv_layers)
    local_rows = [row for row in sorted(local_payload["rows"], key=lambda row: row["layer"]) if row["layer"] in set(mlp_layers)]

    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.float32)
    model.eval()

    calibration_prompts = load_prompts(args.calibration_prompts, [])
    eval_prompts = load_prompts(args.eval_prompts, EVAL_PROMPTS)
    cal_batch = tokenize(tokenizer, calibration_prompts, args.max_length)
    eval_batch = tokenize(tokenizer, eval_prompts, args.max_length)
    baseline = evaluate_model(model, eval_batch, topk=args.topk)
    baseline_logits = baseline.pop("logits")

    plus_params = build_layer_params(args, model, cal_batch, local_rows, "best")
    plain_params = build_layer_params(args, model, cal_batch, local_rows, "plain")
    plus_by_layer = {item["layer"]: dict(item, primitive="plus") for item in plus_params}
    plain_by_layer = {item["layer"]: dict(item, primitive="plain") for item in plain_params}

    kv_specs = make_kv_specs(kv_layers, args.kv_rotation)
    current = [plain_by_layer[layer] for layer in mlp_layers]
    current_metrics = run_stack(model, eval_batch, baseline_logits, baseline, args, kv_specs, current)
    print(f"swap base plain kl={current_metrics['kl_from_baseline']:.6f} topk={current_metrics['topk_overlap']:.6f}")

    steps = []
    remaining = set(mlp_layers)
    for step_idx in range(args.max_swaps):
        candidates = []
        for layer in sorted(remaining):
            trial = [plus_by_layer[item["layer"]] if item["layer"] == layer else item for item in current]
            metrics = run_stack(model, eval_batch, baseline_logits, baseline, args, kv_specs, trial)
            candidate = {
                "layer": layer,
                "metrics": metrics,
                "delta_vs_current": {
                    "kl": metrics["kl_from_baseline"] - current_metrics["kl_from_baseline"],
                    "topk_overlap": metrics["topk_overlap"] - current_metrics["topk_overlap"],
                    "loss_delta_from_baseline": metrics["loss_delta_from_baseline"]
                    - current_metrics["loss_delta_from_baseline"],
                },
                "params": serializable_params(trial),
            }
            candidates.append(candidate)
            print(
                f"swap try layer={layer} kl={metrics['kl_from_baseline']:.6f} "
                f"delta={candidate['delta_vs_current']['kl']:.6f}"
            )

        best = min(candidates, key=lambda item: item["metrics"]["kl_from_baseline"])
        steps.append({"step": step_idx + 1, "best": best, "candidates": candidates})
        if best["metrics"]["kl_from_baseline"] >= current_metrics["kl_from_baseline"] - args.min_improvement:
            if args.pair_search and len(remaining) >= 2:
                pair_candidates = []
                for left, right in itertools.combinations(sorted(remaining), 2):
                    trial = [
                        plus_by_layer[item["layer"]] if item["layer"] in {left, right} else item for item in current
                    ]
                    metrics = run_stack(model, eval_batch, baseline_logits, baseline, args, kv_specs, trial)
                    candidate = {
                        "layers": [left, right],
                        "metrics": metrics,
                        "delta_vs_current": {
                            "kl": metrics["kl_from_baseline"] - current_metrics["kl_from_baseline"],
                            "topk_overlap": metrics["topk_overlap"] - current_metrics["topk_overlap"],
                            "loss_delta_from_baseline": metrics["loss_delta_from_baseline"]
                            - current_metrics["loss_delta_from_baseline"],
                        },
                        "params": serializable_params(trial),
                    }
                    pair_candidates.append(candidate)
                    print(
                        f"swap try_pair layers={left},{right} kl={metrics['kl_from_baseline']:.6f} "
                        f"delta={candidate['delta_vs_current']['kl']:.6f}"
                    )
                best_pair = min(pair_candidates, key=lambda item: item["metrics"]["kl_from_baseline"])
                steps.append({"step": f"{step_idx + 1}_pair", "best_pair": best_pair, "pair_candidates": pair_candidates})
                if best_pair["metrics"]["kl_from_baseline"] < current_metrics["kl_from_baseline"] - args.min_improvement:
                    chosen = set(best_pair["layers"])
                    current = [plus_by_layer[item["layer"]] if item["layer"] in chosen else item for item in current]
                    current_metrics = best_pair["metrics"]
                    remaining -= chosen
                    print(f"swap accept_pair layers={best_pair['layers']} kl={current_metrics['kl_from_baseline']:.6f}")
                    continue
            print("swap stop no_improvement")
            break

        layer = best["layer"]
        current = [plus_by_layer[item["layer"]] if item["layer"] == layer else item for item in current]
        current_metrics = best["metrics"]
        remaining.remove(layer)
        print(f"swap accept layer={layer} kl={current_metrics['kl_from_baseline']:.6f}")

    return {
        "experiment": "fused_primitive_swap_probe",
        "repo": args.repo,
        "local_result": str(args.local_result),
        "kv_layers": kv_layers,
        "kv_rotation": args.kv_rotation,
        "mlp_layers": mlp_layers,
        "baseline_eval": baseline,
        "initial_plain_metrics": steps[0]["candidates"][0]["metrics"] if False else None,
        "final_metrics": current_metrics,
        "final_params": serializable_params(current),
        "steps": steps,
        "summary": {
            "final_kl": current_metrics["kl_from_baseline"],
            "final_topk": current_metrics["topk_overlap"],
            "final_loss_delta": current_metrics["loss_delta_from_baseline"],
            "accepted_swaps": [
                step["best"]["layer"]
                for step in steps
                if "best" in step and step["best"]["delta_vs_current"]["kl"] < -args.min_improvement
            ],
            "accepted_pair_swaps": [
                step["best_pair"]["layers"]
                for step in steps
                if "best_pair" in step and step["best_pair"]["delta_vs_current"]["kl"] < -args.min_improvement
            ],
        },
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_fused_primitive_swap_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy primitive swap probe for a fixed fused frontier")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--local-result", type=Path, required=True)
    parser.add_argument("--kv-layers", required=True)
    parser.add_argument("--kv-rotation", default="hadamard")
    parser.add_argument("--kv-bits", type=int, default=3)
    parser.add_argument("--kv-alpha", type=float, default=0.75)
    parser.add_argument("--mlp-layers", required=True)
    parser.add_argument("--mlp-bits", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--calibration-prompts", required=True)
    parser.add_argument("--eval-prompts")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-swaps", type=int, default=9)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--pair-search", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    result = run_probe(args)
    path = save_result(result, args.out_dir)
    print(json.dumps({"experiment": result["experiment"], "summary": result["summary"]}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
