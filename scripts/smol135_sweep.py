import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import (
    DEFAULT_REPO,
    apply_value_o_edit,
    apply_vproj_typical_edit,
    apply_vproj_uniform_edit,
    capture_linear_inputs,
    evaluate_model,
    get_attention_shape,
    get_layer,
    restore_weights,
    tokenize,
)


CALIBRATION_PROMPTS = [
    "Write a concise explanation of why entropy matters in compression.",
    "The recipe begins by heating olive oil in a pan, then adding",
    "In Python, a dictionary maps keys to values and can be updated by",
    "A proof by contradiction starts by assuming the opposite of what we want",
    "The spacecraft adjusted its trajectory after the navigation system detected",
    "Translate to French: The small model learns from careful data.",
    "When comparing two algorithms, the most important tradeoff is usually",
    "A database index can speed up queries because it",
    "The ancient library kept records of grain, taxes, and astronomical observations.",
    "If the train leaves at 9:15 and arrives 47 minutes later, it arrives at",
    "Explain the difference between precision and recall in one paragraph.",
    "A secure password manager should store secrets using",
    "Complete the sequence: 2, 4, 8, 16,",
    "The function should return None when the input list is empty because",
    "In a transformer, the residual stream carries information between",
    "The contract was revised after both parties agreed to",
    "Summarize the paragraph in one sentence: careful measurement prevents false progress.",
    "The capital city question is easy when the fact is common, but harder when",
    "A poem about winter might mention frost, silence, and",
    "The model should refuse a request when it involves",
    "For each item in the array, compute the square and append it to",
    "The patient reported mild symptoms after taking the medication, including",
    "A chess player often controls the center before launching",
    "The theorem follows from applying the chain rule twice and simplifying",
]


EVAL_PROMPTS = [
    "Explain why a smaller coordinate system can make quantization less damaging.",
    "Write a JavaScript function that filters even numbers from an array.",
    "The museum exhibit described how early sailors navigated using",
    "If Maya buys 12 notebooks and gives 5 away, she has",
    "A good regression test catches behavior changes by comparing",
    "Translate to Spanish: The experiment produced a surprising result.",
    "The HTTP status code 404 usually means",
    "In machine learning, overfitting happens when",
    "The detective noticed that the timestamp did not match",
    "Complete the analogy: hand is to glove as foot is to",
    "When water freezes, its molecules arrange into",
    "A bash script can read command-line arguments using",
    "Summarize this idea: store repeated operators once and reference them cheaply.",
    "The prime numbers less than ten are",
    "A careful medical answer should mention uncertainty and recommend",
    "The character apologized because they realized",
]


def parse_int_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def parse_float_list(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def parse_str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def candidate_without_logits(metrics: dict) -> dict:
    metrics = dict(metrics)
    metrics.pop("logits", None)
    return metrics


def add_deltas(candidate: dict, baseline: dict) -> dict:
    out = dict(candidate)
    out["loss_delta_from_baseline"] = out["loss"] - baseline["loss"]
    return out


def evaluate_edit(model, eval_batch, baseline_logits, baseline_metrics, topk, apply_fn):
    saved = apply_fn()
    metrics = candidate_without_logits(evaluate_model(model, eval_batch, baseline_logits, topk))
    metrics = add_deltas(metrics, baseline_metrics)
    restore_weights(saved)
    return metrics


def evaluate_edit_pair(
    model,
    cal_batch,
    eval_batch,
    cal_baseline_logits,
    eval_baseline_logits,
    cal_baseline_metrics,
    eval_baseline_metrics,
    topk,
    apply_fn,
):
    saved = apply_fn()
    cal_metrics = candidate_without_logits(evaluate_model(model, cal_batch, cal_baseline_logits, topk))
    cal_metrics = add_deltas(cal_metrics, cal_baseline_metrics)
    eval_metrics = candidate_without_logits(evaluate_model(model, eval_batch, eval_baseline_logits, topk))
    eval_metrics = add_deltas(eval_metrics, eval_baseline_metrics)
    restore_weights(saved)
    return {"calibration": cal_metrics, "eval": eval_metrics}


def load_model_and_batches(args):
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.float32)
    model.eval()
    cal_batch = tokenize(tokenizer, CALIBRATION_PROMPTS, args.max_length)
    eval_batch = tokenize(tokenizer, EVAL_PROMPTS, args.max_length)
    cal_baseline = evaluate_model(model, cal_batch, topk=args.topk)
    cal_baseline_logits = cal_baseline.pop("logits")
    eval_baseline = evaluate_model(model, eval_batch, topk=args.topk)
    eval_baseline_logits = eval_baseline.pop("logits")
    return (
        model,
        cal_batch,
        eval_batch,
        cal_baseline,
        cal_baseline_logits,
        eval_baseline,
        eval_baseline_logits,
    )


def run_value_o_sweep(
    args,
    model,
    cal_batch,
    eval_batch,
    cal_baseline,
    cal_baseline_logits,
    eval_baseline,
    eval_baseline_logits,
) -> dict:
    _, _, num_kv_heads, _ = get_attention_shape(model)
    layers = parse_int_list(args.layers)
    rotations = parse_str_list(args.rotations)
    rows = []

    for layer in layers:
        for kv_head in range(num_kv_heads):
            candidates = {}
            for idx, rotation in enumerate(rotations):
                seed = args.seed + layer * 1009 + kv_head * 97 + idx
                candidates[rotation] = evaluate_edit_pair(
                    model,
                    cal_batch,
                    eval_batch,
                    cal_baseline_logits,
                    eval_baseline_logits,
                    cal_baseline,
                    eval_baseline,
                    args.topk,
                    lambda layer=layer, kv_head=kv_head, rotation=rotation, seed=seed: apply_value_o_edit(
                        model, layer, kv_head, args.bits, args.group_size, rotation, seed
                    ),
                )
            identity_eval = candidates["identity"]["eval"]
            best_name, best_pair = min(
                candidates.items(), key=lambda item: item[1]["calibration"]["kl_from_baseline"]
            )
            best_eval = best_pair["eval"]
            rows.append(
                {
                    "layer": layer,
                    "kv_head": kv_head,
                    "candidates": candidates,
                    "best_by_calibration_kl": best_name,
                    "best_minus_identity": {
                        "kl": best_eval["kl_from_baseline"] - identity_eval["kl_from_baseline"],
                        "logit_mse": best_eval["logit_mse"] - identity_eval["logit_mse"],
                        "topk_overlap": best_eval["topk_overlap"] - identity_eval["topk_overlap"],
                        "abs_loss_delta": abs(best_eval["loss_delta_from_baseline"])
                        - abs(identity_eval["loss_delta_from_baseline"]),
                    },
                }
            )
            print(
                f"value_o layer={layer} kv={kv_head} "
                f"identity_eval_kl={identity_eval['kl_from_baseline']:.6f} "
                f"best_cal={best_name} best_eval_kl={best_eval['kl_from_baseline']:.6f}"
            )

    kl_wins = sum(1 for row in rows if row["best_minus_identity"]["kl"] < 0)
    mse_wins = sum(1 for row in rows if row["best_minus_identity"]["logit_mse"] < 0)
    topk_wins = sum(1 for row in rows if row["best_minus_identity"]["topk_overlap"] > 0)
    best_counts = {
        rotation: sum(1 for row in rows if row["best_by_calibration_kl"] == rotation)
        for rotation in rotations
    }
    return {
        "experiment": "value_o_sweep",
        "bits": args.bits,
        "group_size": args.group_size,
        "layers": layers,
        "rotations": rotations,
        "count": len(rows),
        "summary": {
            "kl_wins": kl_wins,
            "logit_mse_wins": mse_wins,
            "topk_overlap_wins": topk_wins,
            "best_counts": best_counts,
            "mean_kl_delta": sum(row["best_minus_identity"]["kl"] for row in rows) / len(rows),
            "mean_logit_mse_delta": sum(row["best_minus_identity"]["logit_mse"] for row in rows) / len(rows),
            "mean_topk_overlap_delta": sum(row["best_minus_identity"]["topk_overlap"] for row in rows)
            / len(rows),
        },
        "rows": rows,
    }


def run_typical_sweep(args, model, cal_batch, eval_batch, baseline, baseline_logits) -> dict:
    layers = parse_int_list(args.layers)
    top_fracs = parse_float_list(args.top_fracs)
    rows = []

    for layer_idx in layers:
        layer = get_layer(model, layer_idx)
        activations = capture_linear_inputs(model, layer.self_attn.v_proj, cal_batch)
        uniform = evaluate_edit(
            model,
            eval_batch,
            baseline_logits,
            baseline,
            args.topk,
            lambda layer_idx=layer_idx: apply_vproj_uniform_edit(
                model, layer_idx, args.bits, args.group_size
            ),
        )

        for top_frac in top_fracs:
            saved, info = apply_vproj_typical_edit(
                model,
                layer_idx=layer_idx,
                bits_hi=args.typical_hi_bits,
                bits_lo=args.typical_lo_bits,
                top_frac=top_frac,
                group_size=args.group_size,
                activations=activations,
            )
            typical = candidate_without_logits(evaluate_model(model, eval_batch, baseline_logits, args.topk))
            typical = add_deltas(typical, baseline)
            restore_weights(saved)
            rows.append(
                {
                    "layer": layer_idx,
                    "top_frac": top_frac,
                    "uniform": uniform,
                    "typical": {**info, **typical},
                    "typical_minus_uniform": {
                        "kl": typical["kl_from_baseline"] - uniform["kl_from_baseline"],
                        "logit_mse": typical["logit_mse"] - uniform["logit_mse"],
                        "topk_overlap": typical["topk_overlap"] - uniform["topk_overlap"],
                        "abs_loss_delta": abs(typical["loss_delta_from_baseline"])
                        - abs(uniform["loss_delta_from_baseline"]),
                    },
                }
            )
            print(
                f"typical layer={layer_idx} top_frac={top_frac:.3f} "
                f"uniform_kl={uniform['kl_from_baseline']:.6f} "
                f"typical_kl={typical['kl_from_baseline']:.6f} "
                f"eff_bits={info['effective_bits_per_input_direction']:.3f}"
            )

    kl_wins = sum(1 for row in rows if row["typical_minus_uniform"]["kl"] < 0)
    mse_wins = sum(1 for row in rows if row["typical_minus_uniform"]["logit_mse"] < 0)
    topk_wins = sum(1 for row in rows if row["typical_minus_uniform"]["topk_overlap"] > 0)
    return {
        "experiment": "typical_vproj_sweep",
        "uniform_bits": args.bits,
        "typical_hi_bits": args.typical_hi_bits,
        "typical_lo_bits": args.typical_lo_bits,
        "group_size": args.group_size,
        "layers": layers,
        "top_fracs": top_fracs,
        "count": len(rows),
        "summary": {
            "kl_wins": kl_wins,
            "logit_mse_wins": mse_wins,
            "topk_overlap_wins": topk_wins,
            "mean_kl_delta": sum(row["typical_minus_uniform"]["kl"] for row in rows) / len(rows),
            "mean_logit_mse_delta": sum(row["typical_minus_uniform"]["logit_mse"] for row in rows)
            / len(rows),
            "mean_topk_overlap_delta": sum(row["typical_minus_uniform"]["topk_overlap"] for row in rows)
            / len(rows),
        },
        "rows": rows,
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_sweep_{payload['experiment']}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out Smol135 compression sweeps")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--experiment", choices=["value_o", "typical_vproj", "both"], default="both")
    parser.add_argument("--layers", default="0,1,2,4,8,12,16,20,24,29")
    parser.add_argument("--top-fracs", default="0.05,0.10,0.20")
    parser.add_argument("--rotations", default="identity,hadamard")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--typical-hi-bits", type=int, default=4)
    parser.add_argument("--typical-lo-bits", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    (
        model,
        cal_batch,
        eval_batch,
        cal_baseline,
        cal_baseline_logits,
        eval_baseline,
        eval_baseline_logits,
    ) = load_model_and_batches(args)
    shared = {
        "repo": args.repo,
        "baseline_calibration": cal_baseline,
        "baseline_eval": eval_baseline,
        "calibration_prompt_count": len(CALIBRATION_PROMPTS),
        "eval_prompt_count": len(EVAL_PROMPTS),
        "max_length": args.max_length,
    }

    outputs = []
    if args.experiment in {"value_o", "both"}:
        outputs.append(
            shared
            | run_value_o_sweep(
                args,
                model,
                cal_batch,
                eval_batch,
                cal_baseline,
                cal_baseline_logits,
                eval_baseline,
                eval_baseline_logits,
            )
        )
    if args.experiment in {"typical_vproj", "both"}:
        outputs.append(
            shared
            | run_typical_sweep(
                args, model, cal_batch, eval_batch, eval_baseline, eval_baseline_logits
            )
        )

    for output in outputs:
        path = save_result(output, args.out_dir)
        print(json.dumps({"experiment": output["experiment"], "summary": output["summary"]}, indent=2))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
