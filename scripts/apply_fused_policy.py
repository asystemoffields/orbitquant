import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import DEFAULT_REPO, get_layer, tokenize
from smol135_fused_choice_allocator import patch_kv_specs
from smol135_mlp_activation_rotation_probe import capture_mlp_intermediate
from smol135_mlp_hadamard_plus_impact_sweep import patch_mlp_precomputed
from smol135_mlp_hadamard_plus_probe import load_prompts, make_rotation
from smol135_mlp_stack_probe import restore_forwards


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def kv_specs_from_config(config: dict) -> list[dict]:
    return [
        {
            "layer": item["layer"],
            "rotation": item["rotation"],
            "bits": item["bits"],
            "alpha": item["alpha"],
        }
        for item in config["selected_policy"]["kv_layers"]
    ]


def build_mlp_params_from_config(model, tokenizer, config: dict, calibration_prompts: list[str] | None = None) -> list[dict]:
    eval_config = config.get("evaluation", {})
    policy = config["selected_policy"]
    max_length = int(eval_config.get("max_length", 64))
    prompt_path = eval_config.get("calibration_prompts")
    prompts = calibration_prompts or load_prompts(prompt_path, [])
    if not prompts:
        raise ValueError("MLP Hadamard-plus policy requires calibration prompts to reconstruct rotations.")

    batch = move_batch(tokenize(tokenizer, prompts, max_length), model_device(model))
    seed = int(config.get("seed", 1234))
    params = []
    for item in policy["mlp_choices"]:
        layer_idx = int(item["layer"])
        layer = get_layer(model, layer_idx)
        down_weight = layer.mlp.down_proj.weight.detach().float().cpu()
        cal_z, _ = capture_mlp_intermediate(model, batch, layer_idx)
        rotation_name = item["rotation"]
        rotation = make_rotation(
            rotation_name,
            cal_z,
            down_weight,
            int(item["block_size"]),
            seed + layer_idx * 1009 + len(rotation_name),
        )
        params.append(
            {
                "layer": layer_idx,
                "primitive": item["primitive"],
                "rotation": rotation_name,
                "alpha": float(item["alpha"]),
                "rotation_tensor": rotation,
            }
        )
    return params


def apply_fused_policy(model, tokenizer, config: dict, calibration_prompts: list[str] | None = None) -> list:
    """Patch a loaded model with the selected runtime activation-compression policy.

    Returns restore handles. Call `restore_forwards(handles)` to undo the patch.
    """
    handles = []
    kv_specs = kv_specs_from_config(config)
    for key in sorted({(item["bits"], item["alpha"], item["rotation"]) for item in kv_specs}):
        bits, alpha, rotation = key
        layers = [item["layer"] for item in kv_specs if (item["bits"], item["alpha"], item["rotation"]) == key]
        handles.extend(patch_kv_specs(model, [{"layer": layer, "rotation": rotation} for layer in layers], bits, alpha, 1234))

    mlp_choices = config["selected_policy"]["mlp_choices"]
    if mlp_choices:
        bits_by_layer = {item["bits"] for item in mlp_choices}
        if len(bits_by_layer) != 1:
            raise ValueError("Mixed MLP bit widths are not supported by the current runtime helper.")
        mlp_params = build_mlp_params_from_config(model, tokenizer, config, calibration_prompts)
        handles.extend(patch_mlp_precomputed(model, mlp_params, bits_by_layer.pop()))
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and apply a fused activation-compression policy artifact")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--prompt", default="A careful compression experiment should preserve")
    args = parser.parse_args()

    config = load_config(args.config)
    repo = args.repo or config.get("base_model") or DEFAULT_REPO
    print(json.dumps({"repo": repo, "policy": config["selected_policy"], "evaluation": config["evaluation"]}, indent=2))
    if args.dry_run:
        return

    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32)
    model.eval()
    handles = apply_fused_policy(model, tokenizer, config)
    batch = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**batch, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    restore_forwards(handles)


if __name__ == "__main__":
    main()
