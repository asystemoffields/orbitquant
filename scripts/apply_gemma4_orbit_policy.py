from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mlp_choice_to_string(item: dict) -> str:
    return f"{item['layer']}:{item['primitive']}:{item['rotation']}:{item['alpha']}"


def args_from_config(config: dict, calib_max_length: int | None = None) -> SimpleNamespace:
    policy = config["selected_policy"]
    evaluation = config.get("evaluation", {})
    kv_layers = policy["kv_layers"]
    mlp_choices = policy["mlp_choices"]
    first_kv = kv_layers[0] if kv_layers else {"bits": 3, "rotation": "hadamard", "alpha": 0.75}
    first_mlp = mlp_choices[0] if mlp_choices else {"bits": 2, "alpha": 0.375, "block_size": 512}
    return SimpleNamespace(
        kv_layers=",".join(str(item["layer"]) for item in kv_layers),
        kv_bits=int(first_kv["bits"]),
        kv_rotation=first_kv["rotation"],
        kv_alpha=float(first_kv["alpha"]),
        mlp_choices=",".join(mlp_choice_to_string(item) for item in mlp_choices),
        mlp_bits=int(first_mlp["bits"]),
        mlp_alpha=float(first_mlp["alpha"]),
        mlp_block_size=int(first_mlp["block_size"]),
        calib_max_length=int(calib_max_length or evaluation.get("calib_max_length") or 192),
    )


def apply_gemma4_orbit_policy(model, tokenizer, config: dict, calibration_prompts: list[str]) -> list:
    """Patch a loaded Gemma4 model with the selected OrbitQuant runtime policy.

    Returns restore handles. Import `restore_forwards` from
    `gemma4_pmra_orbit_stack_eval` to undo the patch.
    """
    from gemma4_pmra_orbit_stack_eval import apply_orbitquant

    args = args_from_config(config)
    return apply_orbitquant(model, tokenizer, calibration_prompts, args, mode="full")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or apply a Gemma4 OrbitQuant policy artifact.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    print(
        json.dumps(
            {
                "base_model": config.get("base_model"),
                "static_weight_state": config.get("static_weight_state"),
                "selected_policy": config["selected_policy"],
                "evaluation": config.get("evaluation", {}),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return
    raise SystemExit("Import apply_gemma4_orbit_policy() from this module to patch an already loaded Gemma4 model.")


if __name__ == "__main__":
    main()
