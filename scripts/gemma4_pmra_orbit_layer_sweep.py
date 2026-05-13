from __future__ import annotations

import argparse
import json
import sys
import time
from copy import copy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from evaluate_pmra_public_dataset import load_public_prompts, source_payload_from_result
from mlp_codebook_model_forward_gate import evaluate_model, strip_logits
from production_mixed_rate_transcoder_gate import (
    build_tensor_specs,
    filter_specs_for_model,
    group_specs,
    load_model_for_profile,
    open_hf_tensor_source,
    parse_source_specs,
    source_readers,
    total_weight_count,
)

from gemma4_pmra_orbit_stack_eval import (
    SMOL135_14BUS_TO_GEMMA4,
    apply_orbitquant,
    force_eager_attention,
    parse_int_list,
    parse_mlp_choices,
    patch_static_variant,
    restore_forwards,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


def mlp_choice_to_string(item: dict) -> str:
    return f"{item['layer']}:{item['primitive']}:{item['rotation']}:{item['alpha']}"


def clone_args_for_candidate(args, kv_layers: str = "", mlp_choices: str = ""):
    candidate_args = copy(args)
    candidate_args.kv_layers = kv_layers
    candidate_args.mlp_choices = mlp_choices
    return candidate_args


def evaluate_candidate(
    model,
    tokenizer,
    eval_prompts: list[str],
    calib_prompts: list[str],
    fp_last_logits,
    pmra_nll: float,
    candidate_args,
    mode: str,
) -> dict:
    handles = []
    try:
        handles, audit = apply_orbitquant(model, tokenizer, calib_prompts, candidate_args, mode)
        eval_args = SimpleNamespace(eval_max_length=candidate_args.eval_max_length)
        result = strip_logits(evaluate_model(model, tokenizer, eval_prompts, eval_args, fp_last_logits))
        result["delta_nll_vs_pmra"] = float(result["nll"] - pmra_nll)
        result["orbit_audit"] = audit
        return result
    finally:
        restore_forwards(handles)


def run(args) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    result["args"]["layers"] = [int(layer) for layer in result["args"]["layers"]]
    tensor_profile = result["args"].get("tensor_profile", "gemma4")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = load_model_for_profile(args.model_dir, tensor_profile, torch.device(args.device))
    force_eager_attention(model)

    source_paths = parse_source_specs(args.source)
    readers = source_readers(source_paths)
    specs, skipped_specs = filter_specs_for_model(
        model,
        build_tensor_specs(result["args"]["layers"], result["args"]["group_mode"], tensor_profile),
        log_prefix="[orbit-layer]",
    )
    groups = group_specs(specs)

    dataset_config = None if args.dataset_config.lower() in {"", "none", "null"} else args.dataset_config
    eval_prompts, eval_audit = load_public_prompts(
        tokenizer,
        args.dataset,
        dataset_config,
        args.split,
        args.text_column,
        args.prompt_count,
        args.prompt_seed,
        args.eval_max_length,
        args.min_tokens,
    )
    calib_prompts, calib_audit = load_public_prompts(
        tokenizer,
        args.dataset,
        dataset_config,
        args.calib_split,
        args.text_column,
        args.calib_prompt_count,
        args.calib_prompt_seed,
        args.calib_max_length,
        args.min_tokens,
    )

    kv_layers = parse_int_list(args.kv_layers)
    mlp_choices = parse_mlp_choices(args.mlp_choices, args.mlp_alpha)
    sides = {part.strip() for part in args.sides.split(",") if part.strip()}
    unknown_sides = sides - {"kv", "mlp"}
    if unknown_sides:
        raise ValueError(f"unknown sides: {sorted(unknown_sides)}")

    output = {
        "created_utc": datetime.now(UTC).isoformat(),
        "experiment": "gemma4_pmra_orbit_layer_sweep",
        "args": {
            "model_dir": str(args.model_dir),
            "hf": str(args.hf),
            "result_json": str(args.result_json),
            "pmra_variant": args.pmra_variant,
            "sides": sorted(sides),
            "device": args.device,
        },
        "eval_prompt_audit": eval_audit,
        "calibration_prompt_audit": calib_audit,
        "skipped_model_tensors": [spec.logical_name for spec in skipped_specs],
        "baseline": {},
        "rows": [],
    }

    with open_hf_tensor_source(args.hf) as hf:
        total_weights = total_weight_count(hf, model, specs)
        low_payload = source_payload_from_result(result, result["args"]["low_source"])
        payload_bytes = int(low_payload + sum(int(row["extra_bytes"]) for row in result["selections"][args.pmra_variant]))

        print("[orbit-layer] evaluating fp16 reference", flush=True)
        patch_static_variant(model, hf, readers, groups, specs, result, "fp16", tensor_profile)
        eval_args = SimpleNamespace(eval_max_length=args.eval_max_length)
        fp_eval = evaluate_model(model, tokenizer, eval_prompts, eval_args)
        fp_last_logits = fp_eval["captured_last_logits"]
        fp_nll = float(fp_eval["nll"])
        output["baseline"]["fp16"] = strip_logits(fp_eval) | {
            "payload_bytes": int(total_weights * 2),
            "payload_bpw": 16.0,
        }

        print(f"[orbit-layer] evaluating PMRA baseline {args.pmra_variant}", flush=True)
        patch_static_variant(model, hf, readers, groups, specs, result, args.pmra_variant, tensor_profile)
        pmra_eval = strip_logits(evaluate_model(model, tokenizer, eval_prompts, eval_args, fp_last_logits))
        pmra_nll = float(pmra_eval["nll"])
        pmra_eval["delta_nll_vs_fp16"] = float(pmra_nll - fp_nll)
        pmra_eval["payload_bytes"] = payload_bytes
        pmra_eval["payload_bpw"] = float(payload_bytes * 8 / total_weights)
        output["baseline"]["pmra"] = pmra_eval

        if "kv" in sides:
            for layer in kv_layers:
                print(f"[orbit-layer] KV candidate layer {layer}", flush=True)
                candidate_args = clone_args_for_candidate(args, kv_layers=str(layer), mlp_choices="")
                row = evaluate_candidate(
                    model,
                    tokenizer,
                    eval_prompts,
                    calib_prompts,
                    fp_last_logits,
                    pmra_nll,
                    candidate_args,
                    "kv",
                )
                row.update({"side": "kv", "layer": layer, "candidate": f"kv:{layer}"})
                output["rows"].append(row)
                (args.output_dir / "partial_result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

        if "mlp" in sides:
            for item in mlp_choices:
                layer = int(item["layer"])
                print(f"[orbit-layer] MLP candidate layer {layer}", flush=True)
                candidate_args = clone_args_for_candidate(args, kv_layers="", mlp_choices=mlp_choice_to_string(item))
                row = evaluate_candidate(
                    model,
                    tokenizer,
                    eval_prompts,
                    calib_prompts,
                    fp_last_logits,
                    pmra_nll,
                    candidate_args,
                    "mlp",
                )
                row.update(
                    {
                        "side": "mlp",
                        "layer": layer,
                        "candidate": f"mlp:{mlp_choice_to_string(item)}",
                        "primitive": item["primitive"],
                        "rotation": item["rotation"],
                        "alpha": float(item["alpha"]),
                    }
                )
                output["rows"].append(row)
                (args.output_dir / "partial_result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    output["rows_by_delta_desc"] = sorted(output["rows"], key=lambda row: row["delta_nll_vs_pmra"], reverse=True)
    output["rows_by_delta_asc"] = sorted(output["rows"], key=lambda row: row["delta_nll_vs_pmra"])
    output["completed_utc"] = datetime.now(UTC).isoformat()
    return output


def make_markdown(payload: dict) -> str:
    lines = [
        "# Result Card - Gemma4 PMRA OrbitQuant Layer Sweep",
        "",
        "## Baseline",
        "",
        f"- FP16 NLL: `{payload['baseline']['fp16']['nll']:.6f}`",
        f"- PMRA NLL: `{payload['baseline']['pmra']['nll']:.6f}`",
        "",
        "## Candidates",
        "",
        "| Candidate | NLL | Delta vs PMRA | Top-10 overlap vs FP16 |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["rows_by_delta_desc"]:
        lines.append(
            f"| {row['candidate']} | {row['nll']:.6f} | {row['delta_nll_vs_pmra']:.6f} | "
            f"{row.get('top10_overlap_to_fp16', float('nan')):.3f} |"
        )
    return "\n".join(lines)


def main() -> int:
    default_kv_layers = ",".join(map(str, SMOL135_14BUS_TO_GEMMA4["kv_layers"]))
    default_mlp_choices = ",".join(SMOL135_14BUS_TO_GEMMA4["mlp_choices"])

    parser = argparse.ArgumentParser(description="Sweep individual OrbitQuant buses under PMRA Gemma4 weights.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pmra-variant", default="c2_calib_knapsack_mixed")
    parser.add_argument("--sides", default="kv,mlp")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--prompt-count", type=int, default=64)
    parser.add_argument("--calib-prompt-count", type=int, default=16)
    parser.add_argument("--prompt-seed", type=int, default=3701)
    parser.add_argument("--calib-prompt-seed", type=int, default=2701)
    parser.add_argument("--eval-max-length", type=int, default=128)
    parser.add_argument("--calib-max-length", type=int, default=128)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kv-layers", default=default_kv_layers)
    parser.add_argument("--kv-bits", type=int, default=3)
    parser.add_argument("--kv-rotation", default="hadamard")
    parser.add_argument("--kv-alpha", type=float, default=0.75)
    parser.add_argument("--mlp-choices", default=default_mlp_choices)
    parser.add_argument("--mlp-bits", type=int, default=2)
    parser.add_argument("--mlp-alpha", type=float, default=0.375)
    parser.add_argument("--mlp-block-size", type=int, default=512)
    args = parser.parse_args()

    start = time.time()
    payload = run(args)
    payload["wall_time_seconds"] = float(time.time() - start)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "result.md").write_text(make_markdown(payload), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "baseline": payload["baseline"], "rows": payload["rows"]}, indent=2))
    print(f"[orbit-layer] wrote {args.output_dir / 'result.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
