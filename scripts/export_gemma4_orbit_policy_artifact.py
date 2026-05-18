from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_BASE_MODEL = "google/gemma-4-E2B-it"
DEFAULT_NAME = "gemma4-pmra-orbitquant-safe3"
PROFILE_DEFAULTS = {
    "trim64_safe3": {
        "prompt_count": 64,
        "calibration_prompt_count": 16,
        "eval_max_length": 128,
        "calib_max_length": 128,
    },
    "trim128_safe3": {
        "prompt_count": 128,
        "calibration_prompt_count": 24,
        "eval_max_length": 192,
        "calib_max_length": 192,
    },
    "trim128_safe3_folded": {
        "prompt_count": 128,
        "calibration_prompt_count": 24,
        "eval_max_length": 192,
        "calib_max_length": 192,
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_variant(result: dict, name: str) -> dict:
    variants = result.get("variants", {})
    if name not in variants:
        raise ValueError(f"result is missing variant {name!r}; available: {sorted(variants)}")
    return variants[name]


def make_config(result: dict, source_path: Path, base_model: str) -> dict:
    pmra = require_variant(result, "pmra")
    q3 = require_variant(result, "q3_k_s")
    stack = require_variant(result, "pmra_orbitquant")
    audit = stack.get("orbit_audit")
    if not audit:
        raise ValueError("pmra_orbitquant variant does not include an orbit_audit")
    profile = result.get("profile")
    profile_defaults = PROFILE_DEFAULTS.get(profile, {})

    kv_layers = [
        {
            "layer": int(layer),
            "bits": int(audit["kv_bits"]),
            "rotation": audit["kv_rotation"],
            "alpha": float(audit["kv_alpha"]),
        }
        for layer in audit.get("kv_layers", [])
    ]
    mlp_choices = [
        {
            "layer": int(item["layer"]),
            "bits": int(audit["mlp_bits"]),
            "primitive": item["primitive"],
            "rotation": item["rotation"],
            "alpha": float(item["alpha"]),
            "block_size": int(audit["mlp_block_size"]),
        }
        for item in audit.get("mlp_choices", [])
    ]

    folded = bool(audit.get("mlp_fold_down_proj", False))
    mlp_method = (
        "Rotate Gemma4 MLP intermediate activations before calibrated 2-bit quantization, "
        "then consume the rotated basis with pre-rotated down_proj weights."
        if folded
        else "Rotate Gemma4 MLP intermediate activations before calibrated 2-bit quantization, then invert before down_proj."
    )

    return {
        "format": "gemma4-pmra-orbitquant-policy-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "base_model": base_model,
        "source_result": str(source_path.resolve()),
        "source_profile": profile,
        "static_weight_state": {
            "method": "PMRA",
            "variant": pmra.get("static_base"),
            "payload_bpw": pmra.get("payload_bpw"),
            "payload_bytes": pmra.get("payload_bytes"),
        },
        "selected_policy": {
            "name": "safe3",
            "total_buses": len(kv_layers) + len(mlp_choices),
            "mlp_fold_down_proj": folded,
            "kv_layers": kv_layers,
            "mlp_choices": mlp_choices,
        },
        "evaluation": {
            "tokens": stack.get("tokens"),
            "dataset": "wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "prompt_count": result.get("eval_prompt_audit", {}).get("selected_count")
            or profile_defaults.get("prompt_count"),
            "calibration_prompt_count": result.get("calibration_prompt_audit", {}).get("selected_count")
            or profile_defaults.get("calibration_prompt_count"),
            "eval_max_length": profile_defaults.get("eval_max_length"),
            "calib_max_length": profile_defaults.get("calib_max_length"),
            "pmra_nll": pmra["nll"],
            "q3_k_s_nll": q3["nll"],
            "stack_nll": stack["nll"],
            "delta_nll_vs_pmra": float(stack["nll"] - pmra["nll"]),
            "delta_nll_vs_q3_k_s": float(stack["nll"] - q3["nll"]),
            "last_logit_mse_to_fp16": stack.get("last_logit_mse_to_fp16"),
            "top10_overlap_to_fp16": stack.get("top10_overlap_to_fp16"),
            "runtime_savings_estimate": result.get("runtime_savings_estimate", {}),
        },
        "method": {
            "kv_cache": "Rotate K/V activations with Hadamard before 3-bit scalar quantization.",
            "mlp_intermediate": mlp_method,
            "calibration": "Hadamard-plus MLP choices reconstruct prepermutation order from calibration activations and down-proj weights.",
        },
    }


def make_manifest(config: dict) -> dict:
    eval_info = config["evaluation"]
    savings = eval_info.get("runtime_savings_estimate", {})
    return {
        "artifact_type": "compression_policy",
        "format": config["format"],
        "base_model": config["base_model"],
        "static_weight_state": config["static_weight_state"],
        "selected_policy_name": config["selected_policy"]["name"],
        "selected_total_buses": config["selected_policy"]["total_buses"],
        "stack_nll": eval_info["stack_nll"],
        "delta_nll_vs_pmra": eval_info["delta_nll_vs_pmra"],
        "estimated_saved_mib": savings.get("total_saved_mib"),
    }


def table_rows(items: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, divider]
    for item in items:
        rows.append("| " + " | ".join(str(item.get(column, "")) for column in columns) + " |")
    return "\n".join(rows)


def make_readme(config: dict, manifest: dict) -> str:
    eval_info = config["evaluation"]
    savings = eval_info.get("runtime_savings_estimate", {})
    folded_label = " Folded" if config["selected_policy"].get("mlp_fold_down_proj", False) else ""
    return f"""---
base_model: {config["base_model"]}
library_name: transformers
tags:
  - orbitquant
  - quantization
  - activation-compression
  - pmra
  - gemma4
pipeline_tag: text-generation
---

# Gemma4 PMRA OrbitQuant Safe3{folded_label} Policy

Base model: `{config["base_model"]}`

This artifact records the current Gemma4 OrbitQuant runtime overlay evaluated on top of the PMRA `c2_calib_knapsack_mixed` static weight state.

## Selected Result

| Metric | Value |
|---|---:|
| Total compressed buses | {config["selected_policy"]["total_buses"]} |
| MLP folded down-proj | {str(config["selected_policy"].get("mlp_fold_down_proj", False)).lower()} |
| PMRA NLL | {eval_info["pmra_nll"]:.6f} |
| Stack NLL | {eval_info["stack_nll"]:.6f} |
| Delta NLL vs PMRA | {eval_info["delta_nll_vs_pmra"]:.6f} |
| Delta NLL vs q3_k_s | {eval_info["delta_nll_vs_q3_k_s"]:.6f} |
| Estimated saved MiB | {savings.get("total_saved_mib")} |

## KV Policy

{table_rows(config["selected_policy"]["kv_layers"], ["layer", "bits", "rotation", "alpha"])}

## MLP Policy

{table_rows(config["selected_policy"]["mlp_choices"], ["layer", "bits", "primitive", "rotation", "alpha", "block_size"])}

## Evaluation

Tokens: `{eval_info["tokens"]}`

Prompt count: `{eval_info.get("prompt_count")}`

Calibration prompt count: `{eval_info.get("calibration_prompt_count")}`

Eval max length: `{eval_info.get("eval_max_length")}`

Calibration max length: `{eval_info.get("calib_max_length")}`

Top-10 overlap vs FP16: `{eval_info.get("top10_overlap_to_fp16")}`

Last-logit MSE vs FP16: `{eval_info.get("last_logit_mse_to_fp16")}`

## Files

- `compression_config.json`: runtime policy and metrics.
- `manifest.json`: compact artifact summary.
- `README.md`: model-card draft for publication.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Gemma4 PMRA OrbitQuant policy artifact.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    args = parser.parse_args()

    result = load_json(args.result)
    out_dir = args.out_dir or Path("hf_artifacts") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = make_config(result, args.result, args.base_model)
    manifest = make_manifest(config)
    readme = make_readme(config, manifest)

    (out_dir / "compression_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
