import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import DEFAULT_REPO, evaluate_model, tokenize
from smol135_fused_stack_probe import patch_attention_layers
from smol135_mlp_hadamard_plus_impact_sweep import build_layer_params, load_json, patch_mlp_precomputed
from smol135_mlp_hadamard_plus_probe import load_prompts
from smol135_mlp_stack_probe import restore_forwards, summarize_candidate
from smol135_sweep import EVAL_PROMPTS, parse_int_list


def serializable_params(params: list[dict]) -> list[dict]:
    rows = []
    for item in params:
        rows.append(
            {
                "layer": item["layer"],
                "rotation": item["rotation"],
                "alpha": item["alpha"],
                "local_eval_down_output_rel_mse": item.get("local_eval_down_output_rel_mse"),
            }
        )
    return rows


def run_candidate(
    model,
    eval_batch,
    baseline_logits,
    baseline,
    topk: int,
    kv_layers: list[int],
    kv_bits: int,
    kv_rotation: str,
    kv_alpha: float,
    mlp_params: list[dict],
    mlp_bits: int,
    seed: int,
) -> dict:
    saved_kv = patch_attention_layers(model, kv_layers, kv_bits, kv_rotation, kv_alpha, seed) if kv_layers else []
    saved_mlp = patch_mlp_precomputed(model, mlp_params, mlp_bits) if mlp_params else []
    metrics = summarize_candidate(model, eval_batch, baseline_logits, baseline, topk)
    restore_forwards(saved_mlp)
    restore_forwards(saved_kv)
    return metrics


def run_probe(args) -> dict:
    local_result = load_json(args.local_result)
    mlp_layers = set(parse_int_list(args.mlp_layers))
    local_rows = [row for row in sorted(local_result["rows"], key=lambda row: row["layer"]) if row["layer"] in mlp_layers]

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
    kv_layers = parse_int_list(args.kv_layers)

    specs = [
        ("kv_only", kv_layers, []),
        ("mlp_plus_only", [], plus_params),
        ("mlp_plain_same_layers_only", [], plain_params),
        ("fused_kv_mlp_plus", kv_layers, plus_params),
        ("fused_kv_mlp_plain_same_layers", kv_layers, plain_params),
    ]

    candidates = []
    for name, kv, mlp in specs:
        metrics = run_candidate(
            model,
            eval_batch,
            baseline_logits,
            baseline,
            args.topk,
            kv,
            args.kv_bits,
            args.kv_rotation,
            args.kv_alpha,
            mlp,
            args.mlp_bits,
            args.seed,
        )
        candidates.append({"candidate": name, "metrics": metrics})
        print(
            f"eval {name} kl={metrics['kl_from_baseline']:.6f} "
            f"topk={metrics['topk_overlap']:.6f} loss_delta={metrics['loss_delta_from_baseline']:.6f}"
        )

    by_name = {item["candidate"]: item["metrics"] for item in candidates}
    return {
        "experiment": "fused_hadamard_plus_probe",
        "repo": args.repo,
        "local_result": str(args.local_result),
        "kv_layers": kv_layers,
        "kv_bits": args.kv_bits,
        "kv_rotation": args.kv_rotation,
        "kv_alpha": args.kv_alpha,
        "mlp_layers": sorted(mlp_layers),
        "mlp_bits": args.mlp_bits,
        "block_size": args.block_size,
        "calibration_prompt_count": len(calibration_prompts),
        "eval_prompt_count": len(eval_prompts),
        "baseline_eval": baseline,
        "plus_params": serializable_params(plus_params),
        "plain_params": serializable_params(plain_params),
        "candidates": candidates,
        "summary": {
            "fused_plus_minus_plain_same_layers": {
                "kl": by_name["fused_kv_mlp_plus"]["kl_from_baseline"]
                - by_name["fused_kv_mlp_plain_same_layers"]["kl_from_baseline"],
                "topk_overlap": by_name["fused_kv_mlp_plus"]["topk_overlap"]
                - by_name["fused_kv_mlp_plain_same_layers"]["topk_overlap"],
                "loss_delta_from_baseline": by_name["fused_kv_mlp_plus"]["loss_delta_from_baseline"]
                - by_name["fused_kv_mlp_plain_same_layers"]["loss_delta_from_baseline"],
            },
            "mlp_plus_minus_plain_same_layers": {
                "kl": by_name["mlp_plus_only"]["kl_from_baseline"]
                - by_name["mlp_plain_same_layers_only"]["kl_from_baseline"],
                "topk_overlap": by_name["mlp_plus_only"]["topk_overlap"]
                - by_name["mlp_plain_same_layers_only"]["topk_overlap"],
                "loss_delta_from_baseline": by_name["mlp_plus_only"]["loss_delta_from_baseline"]
                - by_name["mlp_plain_same_layers_only"]["loss_delta_from_baseline"],
            },
        },
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_fused_hadamard_plus_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fused KV + Hadamard-plus MLP stack probe")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--local-result", type=Path, required=True)
    parser.add_argument("--kv-layers", default="28,24,26,14")
    parser.add_argument("--kv-bits", type=int, default=3)
    parser.add_argument("--kv-rotation", default="hadamard")
    parser.add_argument("--kv-alpha", type=float, default=0.75)
    parser.add_argument("--mlp-layers", required=True)
    parser.add_argument("--mlp-bits", type=int, default=2)
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
    print(json.dumps({"experiment": result["experiment"], "summary": result["summary"], "candidates": result["candidates"]}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
