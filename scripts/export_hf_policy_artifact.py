import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def unwrap_result(raw: dict) -> dict:
    payload = raw.get("payload")
    if isinstance(payload, dict):
        return payload
    return raw


def arg_value(args: list[str], name: str, default=None):
    if name not in args:
        return default
    index = args.index(name)
    if index + 1 >= len(args):
        return default
    return args[index + 1]


def choose_policy(source: dict, target_buses: int | None = None, select: str = "best") -> dict:
    summary = source.get("summary", {})
    frontier = summary.get("frontier", [])
    if frontier:
        candidates = frontier
        if target_buses is not None:
            candidates = [item for item in frontier if item["total_buses"] == target_buses]
            if not candidates:
                available = [item["total_buses"] for item in frontier]
                raise ValueError(f"No frontier entry has total_buses={target_buses}. Available: {available}")
        if select == "last":
            return candidates[-1]
        return min(candidates, key=lambda item: (item.get("score", item["kl"]), item["kl"]))

    if "final_params" in source and "final_metrics" in source:
        metrics = source["final_metrics"]
        return {
            "total_buses": len(source["kv_layers"]) + len(source["final_params"]),
            "kv_layers": source["kv_layers"],
            "mlp_choices": [
                {
                    "layer": item["layer"],
                    "primitive": item.get("primitive", "plain"),
                    "rotation": item["rotation"],
                    "alpha": item["alpha"],
                }
                for item in source["final_params"]
            ],
            "kl": metrics["kl_from_baseline"],
            "topk": metrics["topk_overlap"],
            "loss_delta": metrics["loss"] - source["baseline_eval"]["loss"],
            "saved_mb": None,
            "score": metrics["kl_from_baseline"],
        }

    raise ValueError("Could not find a policy frontier in the result JSON.")


def make_config(raw: dict, source: dict, source_path: Path, policy: dict) -> dict:
    args = raw.get("args", [])
    repo = source.get("repo") or DEFAULT_BASE_MODEL
    kv_alpha = arg_value(args, "--kv-alpha", "0.75")
    kv_bits = arg_value(args, "--kv-bits", "3")
    mlp_bits = arg_value(args, "--mlp-bits", "2")
    block_size = arg_value(args, "--block-size", "512")
    max_length = arg_value(args, "--max-length", "64")
    seed = arg_value(args, "--seed", "1234")
    eval_prompts = arg_value(args, "--eval-prompts", "prompts/broad_eval.txt")
    calibration_prompts = arg_value(args, "--calibration-prompts", "prompts/broad_calibration.txt")

    return {
        "format": "smol135-fused-activation-policy-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": repo,
        "source_result": str(source_path.resolve()),
        "profile": raw.get("profile"),
        "seed": int(seed),
        "selected_policy": {
            "total_buses": policy["total_buses"],
            "kv_layers": [
                {
                    "layer": layer,
                    "bits": int(kv_bits),
                    "rotation": "hadamard",
                    "alpha": float(kv_alpha),
                }
                for layer in policy["kv_layers"]
            ],
            "mlp_choices": [
                {
                    "layer": item["layer"],
                    "bits": int(mlp_bits),
                    "primitive": item["primitive"],
                    "rotation": item["rotation"],
                    "alpha": item["alpha"],
                    "block_size": int(block_size),
                }
                for item in policy["mlp_choices"]
            ],
        },
        "evaluation": {
            "kl_from_baseline": policy["kl"],
            "topk_overlap": policy["topk"],
            "loss_delta": policy["loss_delta"],
            "estimated_saved_mb": policy.get("saved_mb"),
            "max_length": int(max_length),
            "calibration_prompts": calibration_prompts,
            "eval_prompts": eval_prompts,
        },
        "method": {
            "kv_cache": "Rotate K/V activations with Hadamard before 3-bit scalar quantization.",
            "mlp_intermediate": "Rotate SwiGLU intermediate activations before calibrated 2-bit quantization, then invert before down_proj.",
            "primitive_choice": "MLP layers may choose plain block-Hadamard or activation/down-proj-aware Hadamard-plus.",
        },
    }


def make_manifest(config: dict, source: dict) -> dict:
    return {
        "artifact_type": "compression_policy",
        "format": config["format"],
        "base_model": config["base_model"],
        "result_profile": config["profile"],
        "selected_total_buses": config["selected_policy"]["total_buses"],
        "kl_from_baseline": config["evaluation"]["kl_from_baseline"],
        "topk_overlap": config["evaluation"]["topk_overlap"],
        "stopped_reason": source.get("stopped_reason") or source.get("summary", {}).get("stopped_reason"),
    }


def table_rows(items: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, divider]
    for item in items:
        rows.append("| " + " | ".join(str(item.get(column, "")) for column in columns) + " |")
    return "\n".join(rows)


def make_readme(config: dict, manifest: dict) -> str:
    kv_rows = [
        {
            "layer": item["layer"],
            "bits": item["bits"],
            "rotation": item["rotation"],
            "alpha": item["alpha"],
        }
        for item in config["selected_policy"]["kv_layers"]
    ]
    mlp_rows = [
        {
            "layer": item["layer"],
            "bits": item["bits"],
            "primitive": item["primitive"],
            "rotation": item["rotation"],
            "alpha": item["alpha"],
            "block_size": item["block_size"],
        }
        for item in config["selected_policy"]["mlp_choices"]
    ]
    eval_info = config["evaluation"]
    return f"""# Smol135 Fused Activation Policy

Base model: `{config["base_model"]}`

This artifact records a fused activation-compression policy for SmolLM2-135M. It packages the selected KV-cache and MLP intermediate-bus transforms, the source result path, and the held-out evaluation metrics used to choose the policy.

## Selected Result

| Metric | Value |
|---|---:|
| Total compressed buses | {config["selected_policy"]["total_buses"]} |
| KL from baseline | {eval_info["kl_from_baseline"]:.6f} |
| Top-k overlap | {eval_info["topk_overlap"]:.6f} |
| Loss delta | {eval_info["loss_delta"]:.6f} |
| Estimated saved MB | {eval_info["estimated_saved_mb"]} |

## KV Policy

{table_rows(kv_rows, ["layer", "bits", "rotation", "alpha"])}

## MLP Policy

{table_rows(mlp_rows, ["layer", "bits", "primitive", "rotation", "alpha", "block_size"])}

## Evaluation

Calibration prompts: `{eval_info["calibration_prompts"]}`

Held-out eval prompts: `{eval_info["eval_prompts"]}`

Max sequence length: `{eval_info["max_length"]}`

Stopped reason: `{manifest.get("stopped_reason")}`

## Files

- `compression_config.json`: runtime policy and metrics.
- `manifest.json`: compact artifact summary.
- `README.md`: model-card draft for publication.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--target-buses", type=int)
    parser.add_argument("--select", choices=["best", "last"], default="best")
    args = parser.parse_args()

    raw = load_json(args.result)
    source = unwrap_result(raw)
    policy = choose_policy(source, target_buses=args.target_buses, select=args.select)
    name = args.name or f"smol135-fused-policy-{args.result.stem}"
    out_dir = args.out_dir or Path("hf_artifacts") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = make_config(raw, source, args.result, policy)
    manifest = make_manifest(config, source)
    readme = make_readme(config, manifest)

    (out_dir / "compression_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
