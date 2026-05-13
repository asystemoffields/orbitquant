import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import DEFAULT_REPO, evaluate_model, tokenize
from smol135_fused_stack_probe import patch_attention_layers, restore_forwards, summarize_candidate
from smol135_sweep import CALIBRATION_PROMPTS, EVAL_PROMPTS, parse_int_list, parse_str_list


def load_prompts(path: str | None, fallback: list[str]) -> list[str]:
    if path is None:
        return fallback
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def eval_kv_candidate(
    model,
    layer: int,
    rotation: str,
    bits: int,
    alpha: float,
    cal_batch,
    eval_batch,
    cal_baseline,
    cal_baseline_logits,
    eval_baseline,
    eval_baseline_logits,
    topk: int,
    seed: int,
) -> dict:
    saved = patch_attention_layers(model, [layer], bits, rotation, alpha, seed)
    cal = summarize_candidate(model, cal_batch, cal_baseline_logits, cal_baseline, topk)
    ev = summarize_candidate(model, eval_batch, eval_baseline_logits, eval_baseline, topk)
    restore_forwards(saved)
    return {"calibration": cal, "eval": ev}


def run_sweep(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.float32)
    model.eval()

    layers = parse_int_list(args.layers)
    rotations = parse_str_list(args.rotations)
    calibration_prompts = load_prompts(args.calibration_prompts, CALIBRATION_PROMPTS)
    eval_prompts = load_prompts(args.eval_prompts, EVAL_PROMPTS)
    cal_batch = tokenize(tokenizer, calibration_prompts, args.max_length)
    eval_batch = tokenize(tokenizer, eval_prompts, args.max_length)
    cal_baseline = evaluate_model(model, cal_batch, topk=args.topk)
    cal_baseline_logits = cal_baseline.pop("logits")
    eval_baseline = evaluate_model(model, eval_batch, topk=args.topk)
    eval_baseline_logits = eval_baseline.pop("logits")

    rows = []
    for layer in layers:
        candidates = {}
        for rotation in rotations:
            candidates[rotation] = eval_kv_candidate(
                model,
                layer,
                rotation,
                args.bits,
                args.alpha,
                cal_batch,
                eval_batch,
                cal_baseline,
                cal_baseline_logits,
                eval_baseline,
                eval_baseline_logits,
                args.topk,
                args.seed + layer * 1009 + len(rotation),
            )
        identity = candidates["identity"]["eval"]
        best_name, best_pair = min(
            candidates.items(), key=lambda item: item[1]["calibration"]["kl_from_baseline"]
        )
        best_eval = best_pair["eval"]
        rows.append(
            {
                "layer": layer,
                "candidates": candidates,
                "best_by_calibration": best_name,
                "best_minus_identity_eval": {
                    "kl": best_eval["kl_from_baseline"] - identity["kl_from_baseline"],
                    "topk_overlap": best_eval["topk_overlap"] - identity["topk_overlap"],
                    "loss_delta_abs": abs(best_eval["loss_delta_from_baseline"])
                    - abs(identity["loss_delta_from_baseline"]),
                },
            }
        )
        print(
            f"kv-impact layer={layer} identity_kl={identity['kl_from_baseline']:.6f} "
            f"best={best_name} best_eval_kl={best_eval['kl_from_baseline']:.6f}"
        )

    stacks = []
    sorted_rows = sorted(rows, key=lambda row: row["candidates"][row["best_by_calibration"]]["eval"]["kl_from_baseline"])
    for count in parse_int_list(args.top_counts):
        selected_rows = sorted_rows[:count]
        selected_layers = [row["layer"] for row in selected_rows]
        # Current stack uses one rotation for all selected layers. If selected layers disagree,
        # prefer Hadamard when present because it is cheap and dominated local KV probes.
        hadamard_layers = [
            row["layer"]
            for row in selected_rows
            if row["best_by_calibration"] == "hadamard"
        ]
        identity_layers = [
            row["layer"]
            for row in selected_rows
            if row["best_by_calibration"] == "identity"
        ]
        random_layers = [
            row["layer"]
            for row in selected_rows
            if row["best_by_calibration"] == "random_orthogonal"
        ]

        saved = []
        if identity_layers:
            saved += patch_attention_layers(model, identity_layers, args.bits, "identity", args.alpha, args.seed)
        if hadamard_layers:
            saved += patch_attention_layers(model, hadamard_layers, args.bits, "hadamard", args.alpha, args.seed)
        if random_layers:
            saved += patch_attention_layers(
                model, random_layers, args.bits, "random_orthogonal", args.alpha, args.seed
            )
        metrics = summarize_candidate(model, eval_batch, eval_baseline_logits, eval_baseline, args.topk)
        restore_forwards(saved)
        stacks.append(
            {
                "top_count": count,
                "layers": selected_layers,
                "rotation_counts": {
                    "identity": len(identity_layers),
                    "hadamard": len(hadamard_layers),
                    "random_orthogonal": len(random_layers),
                },
                "metrics": metrics,
            }
        )
        print(
            f"kv-stack top_count={count} layers={selected_layers} "
            f"kl={metrics['kl_from_baseline']:.6f} topk={metrics['topk_overlap']:.6f}"
        )

    return {
        "experiment": "kv_impact_sweep",
        "repo": args.repo,
        "layers": layers,
        "bits": args.bits,
        "alpha": args.alpha,
        "calibration_prompt_count": len(calibration_prompts),
        "eval_prompt_count": len(eval_prompts),
        "rotations": rotations,
        "baseline_eval": eval_baseline,
        "rows": rows,
        "stacks": stacks,
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_kv_impact_{payload['bits']}bit_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-layer end-to-end KV activation quantization impact sweep")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--rotations", default="identity,hadamard")
    parser.add_argument("--top-counts", default="4,8,12,16")
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
                "best": row["best_by_calibration"],
                "eval_kl": row["candidates"][row["best_by_calibration"]]["eval"]["kl_from_baseline"],
                "eval_topk": row["candidates"][row["best_by_calibration"]]["eval"]["topk_overlap"],
            }
            for row in sorted(
                result["rows"],
                key=lambda row: row["candidates"][row["best_by_calibration"]]["eval"]["kl_from_baseline"],
            )[:10]
        ],
        "stacks": result["stacks"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
