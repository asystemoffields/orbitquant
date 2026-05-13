from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer


def _add_pmra_scripts_to_path() -> None:
    candidates = []
    env_path = os.environ.get("PMRA_SCRIPTS")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path.home() / "Documents" / "PMRA" / "scripts",
            Path("/workspace/pmra/scripts"),
        ]
    )
    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


_add_pmra_scripts_to_path()

from evaluate_pmra_public_dataset import (  # noqa: E402
    load_public_prompts,
    source_payload_from_result,
    variant_payload_bytes,
)
from mlp_codebook_model_forward_gate import (  # noqa: E402
    copy_array_to_parameter,
    evaluate_model,
    strip_logits,
)
from production_mixed_rate_transcoder_gate import (  # noqa: E402
    apply_selection,
    build_tensor_specs,
    filter_specs_for_model,
    group_specs,
    hf_ref_array,
    load_model_for_profile,
    open_hf_tensor_source,
    parse_source_specs,
    patch_all_from_source,
    selected_extra_bytes,
    source_readers,
    total_weight_count,
)

from transformers.models.gemma4 import modeling_gemma4 as gemma4_modeling  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


SMOL135_14BUS_TO_GEMMA4 = {
    "kv_layers": [33, 28, 30, 16, 18, 11, 15],
    "mlp_choices": [
        "19:plus:preperm_activation_max_hadamard:0.375",
        "20:plus:preperm_activation_max_hadamard:0.375",
        "18:plain:block_hadamard:0.375",
        "14:plain:block_hadamard:0.375",
        "6:plus:preperm_boundary_rms_hadamard:0.375",
        "16:plain:block_hadamard:0.375",
        "15:plain:block_hadamard:0.375",
    ],
}


_HADAMARD_CACHE: dict[int, torch.Tensor] = {}


def hadamard(dim: int) -> torch.Tensor:
    if dim <= 0 or dim & (dim - 1):
        raise ValueError(f"Hadamard dimension must be a power of two, got {dim}")
    cached = _HADAMARD_CACHE.get(dim)
    if cached is not None:
        return cached
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < dim:
        h = torch.cat(
            [
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ],
            dim=0,
        )
    h = h / math.sqrt(dim)
    _HADAMARD_CACHE[dim] = h
    return h


def get_text_model(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise AttributeError("could not find the text decoder stack on the loaded model")


def get_text_layer(model, layer_idx: int):
    return get_text_model(model).layers[layer_idx]


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def force_eager_attention(model) -> None:
    configs = [getattr(model, "config", None)]
    if configs[0] is not None:
        text_config = getattr(configs[0], "text_config", None)
        if text_config is not None:
            configs.append(text_config)
        if hasattr(configs[0], "get_text_config"):
            try:
                configs.append(configs[0].get_text_config())
            except Exception:
                pass
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None:
            configs.append(cfg)
    for cfg in configs:
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def tokenize_batch(tokenizer, prompts: list[str], max_length: int, device: torch.device) -> dict:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return move_batch(
        {key: value for key, value in batch.items() if key in {"input_ids", "attention_mask"}},
        device,
    )


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_mlp_choices(value: str, default_alpha: float) -> list[dict]:
    choices = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) == 1:
            layer = int(parts[0])
            primitive = "plain"
            rotation = "block_hadamard"
            alpha = default_alpha
        elif len(parts) in {2, 3, 4}:
            layer = int(parts[0])
            primitive = parts[1] or "plain"
            rotation = parts[2] if len(parts) >= 3 and parts[2] else "block_hadamard"
            alpha = float(parts[3]) if len(parts) == 4 and parts[3] else default_alpha
        else:
            raise ValueError(f"MLP choice must be layer[:primitive[:rotation[:alpha]]], got {raw!r}")
        choices.append(
            {
                "layer": layer,
                "primitive": primitive,
                "rotation": rotation,
                "alpha": alpha,
            }
        )
    return choices


def balanced_order(scores: torch.Tensor, block_size: int) -> list[int]:
    dim = int(scores.numel())
    if dim % block_size:
        raise ValueError(f"dimension {dim} must be divisible by block size {block_size}")
    block_count = dim // block_size
    block_loads = [0.0 for _ in range(block_count)]
    block_slots: list[list[int]] = [[] for _ in range(block_count)]

    ranked = torch.argsort(scores.detach().float().cpu(), descending=True).tolist()
    for channel in ranked:
        available = [idx for idx in range(block_count) if len(block_slots[idx]) < block_size]
        target = min(available, key=lambda idx: (block_loads[idx], len(block_slots[idx]), idx))
        block_slots[target].append(channel)
        block_loads[target] += float(scores[channel])
    return [channel for slots in block_slots for channel in slots]


def build_rotation_spec(
    name: str,
    z: torch.Tensor | None,
    down_weight: torch.Tensor | None,
    dim: int,
    block_size: int,
) -> dict:
    if name == "identity":
        return {"name": name, "dim": dim, "block_size": 0, "order": None}
    if dim % block_size:
        raise ValueError(f"rotation {name} requires dim {dim} divisible by block size {block_size}")
    order = None
    if name.startswith("preperm_"):
        if z is None or down_weight is None:
            raise ValueError(f"rotation {name} needs calibration activations and down-proj weights")
        activation_rms = z.pow(2).mean(dim=0).sqrt()
        activation_max = z.abs().amax(dim=0)
        down_norm = down_weight.norm(dim=0)
        boundary_rms = activation_rms * down_norm
        boundary_max = activation_max * down_norm
        score_map = {
            "preperm_activation_rms_hadamard": activation_rms,
            "preperm_activation_max_hadamard": activation_max,
            "preperm_down_norm_hadamard": down_norm,
            "preperm_boundary_rms_hadamard": boundary_rms,
            "preperm_boundary_max_hadamard": boundary_max,
        }
        if name not in score_map:
            raise ValueError(f"unsupported prepermutation rotation {name!r}")
        order = balanced_order(score_map[name], block_size)
    elif name not in {"block_hadamard", "hadamard"}:
        raise ValueError(f"unsupported rotation {name!r}")
    return {"name": name, "dim": dim, "block_size": block_size, "order": order}


def apply_block_hadamard(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if block_size == 0:
        return x
    dim = x.shape[-1]
    h = hadamard(block_size).to(device=x.device, dtype=x.dtype)
    flat = x.reshape(-1, dim)
    blocks = flat.reshape(-1, dim // block_size, block_size)
    rotated = torch.matmul(blocks, h)
    return rotated.reshape_as(flat).reshape_as(x)


def rotate_forward(x: torch.Tensor, spec: dict) -> torch.Tensor:
    if spec["name"] == "identity":
        return x
    work = x
    if spec.get("order") is not None:
        order = torch.as_tensor(spec["order"], device=x.device, dtype=torch.long)
        work = work.index_select(-1, order)
    return apply_block_hadamard(work, int(spec["block_size"]))


def rotate_inverse(x: torch.Tensor, spec: dict) -> torch.Tensor:
    if spec["name"] == "identity":
        return x
    work = apply_block_hadamard(x, int(spec["block_size"]))
    if spec.get("order") is not None:
        order = torch.as_tensor(spec["order"], device=x.device, dtype=torch.long)
        restored = torch.empty_like(work)
        restored.index_copy_(-1, order, work)
        return restored
    return work


def quantize_last_dim(x: torch.Tensor, bits: int, rotation_spec: dict, alpha: float) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    if qmax <= 0:
        raise ValueError(f"bits must be >= 2, got {bits}")
    original_dtype = x.dtype
    rotated = rotate_forward(x.float(), rotation_spec)
    scale = rotated.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) * alpha / qmax
    quantized = torch.round(rotated / scale).clamp(-qmax, qmax) * scale
    return rotate_inverse(quantized, rotation_spec).to(dtype=original_dtype)


def make_gemma4_attention_forward(attn, bits: int, rotation_spec: dict, alpha: float):
    apply_rotary_pos_emb = gemma4_modeling.apply_rotary_pos_emb
    eager_attention_forward = gemma4_modeling.eager_attention_forward
    attention_functions = gemma4_modeling.ALL_ATTENTION_FUNCTIONS

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        if self.is_kv_shared_layer and past_key_values is not None:
            key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        key_states = quantize_last_dim(key_states, bits, rotation_spec, alpha)
        value_states = quantize_last_dim(value_states, bits, rotation_spec, alpha)

        if past_key_values is not None:
            if not self.is_kv_shared_layer:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            if self.store_full_length_kv:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[self.layer_idx] = key_states, value_states

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = attention_functions[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return types.MethodType(forward, attn)


def patch_kv_layers(model, layers: list[int], bits: int, rotation: str, alpha: float) -> list:
    saved = []
    for layer_idx in layers:
        attn = get_text_layer(model, layer_idx).self_attn
        rotation_spec = build_rotation_spec(
            "block_hadamard" if rotation == "hadamard" else rotation,
            z=None,
            down_weight=None,
            dim=int(attn.head_dim),
            block_size=int(attn.head_dim),
        )
        original_forward = attn.forward
        attn.forward = make_gemma4_attention_forward(attn, bits, rotation_spec, alpha)
        saved.append((attn, original_forward))
    return saved


def capture_mlp_intermediate(model, tokenizer, prompts: list[str], layer_idx: int, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_text_layer(model, layer_idx)
    captured = {}

    def hook(module, inputs, _output):
        x = inputs[0].detach()
        gate = module.act_fn(module.gate_proj(x))
        up = module.up_proj(x)
        z = gate * up
        y = module.down_proj(z)
        captured["z"] = z.reshape(-1, z.shape[-1]).detach().float().cpu()
        captured["y"] = y.reshape(-1, y.shape[-1]).detach().float().cpu()

    handle = layer.mlp.register_forward_hook(hook)
    try:
        batch = tokenize_batch(tokenizer, prompts, max_length, model_device(model))
        with torch.inference_mode():
            model(**batch, use_cache=False)
    finally:
        handle.remove()
    return captured["z"], captured["y"]


def make_mlp_forward(mlp, bits: int, rotation_spec: dict, alpha: float):
    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        z = gate * up
        z_quant = quantize_last_dim(z, bits, rotation_spec, alpha)
        return self.down_proj(z_quant)

    return types.MethodType(forward, mlp)


def build_mlp_runtime_params(
    model,
    tokenizer,
    choices: list[dict],
    calibration_prompts: list[str],
    calib_max_length: int,
    block_size: int,
) -> list[dict]:
    params = []
    for item in choices:
        layer_idx = int(item["layer"])
        layer = get_text_layer(model, layer_idx)
        print(
            f"[orbit-gemma4] calibrating MLP layer {layer_idx} rotation={item['rotation']} alpha={item['alpha']}",
            flush=True,
        )
        down_weight = layer.mlp.down_proj.weight.detach().float().cpu()
        z, _y = capture_mlp_intermediate(model, tokenizer, calibration_prompts, layer_idx, calib_max_length)
        rotation_spec = build_rotation_spec(
            item["rotation"],
            z=z,
            down_weight=down_weight,
            dim=int(z.shape[-1]),
            block_size=block_size,
        )
        serial_spec = {
            "name": rotation_spec["name"],
            "dim": rotation_spec["dim"],
            "block_size": rotation_spec["block_size"],
            "has_order": rotation_spec.get("order") is not None,
        }
        params.append({**item, "rotation_spec": rotation_spec, "rotation_summary": serial_spec})
    return params


def patch_mlp_layers(model, params: list[dict], bits: int) -> list:
    saved = []
    for item in params:
        layer = get_text_layer(model, int(item["layer"]))
        original_forward = layer.mlp.forward
        layer.mlp.forward = make_mlp_forward(
            layer.mlp,
            bits=bits,
            rotation_spec=item["rotation_spec"],
            alpha=float(item["alpha"]),
        )
        saved.append((layer.mlp, original_forward))
    return saved


def restore_forwards(handles: list) -> None:
    for module, original_forward in reversed(handles):
        module.forward = original_forward


def patch_all_from_hf(model, hf, specs) -> None:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for spec in specs:
        arr = hf_ref_array(hf, model, spec)
        if spec.logical_name in params:
            copy_array_to_parameter(params[spec.logical_name], arr)
        elif spec.logical_name in buffers:
            copy_array_to_parameter(buffers[spec.logical_name], arr)


def apply_orbitquant(
    model,
    tokenizer,
    calibration_prompts: list[str],
    args,
) -> tuple[list, dict]:
    kv_layers = parse_int_list(args.kv_layers)
    mlp_choices = parse_mlp_choices(args.mlp_choices, args.mlp_alpha)
    handles = []
    if kv_layers:
        handles.extend(patch_kv_layers(model, kv_layers, args.kv_bits, args.kv_rotation, args.kv_alpha))
    mlp_params = []
    if mlp_choices:
        mlp_params = build_mlp_runtime_params(
            model,
            tokenizer,
            mlp_choices,
            calibration_prompts,
            args.calib_max_length,
            args.mlp_block_size,
        )
        handles.extend(patch_mlp_layers(model, mlp_params, args.mlp_bits))
    audit = {
        "kv_layers": kv_layers,
        "kv_bits": args.kv_bits,
        "kv_rotation": args.kv_rotation,
        "kv_alpha": args.kv_alpha,
        "mlp_bits": args.mlp_bits,
        "mlp_block_size": args.mlp_block_size,
        "mlp_choices": [
            {
                "layer": int(item["layer"]),
                "primitive": item["primitive"],
                "rotation": item["rotation"],
                "alpha": float(item["alpha"]),
                "rotation_summary": item["rotation_summary"],
            }
            for item in mlp_params
        ],
    }
    return handles, audit


def static_payload_bytes(result: dict, variant: str, low_payload: int, total_weights: int) -> int:
    if variant == "fp16":
        return int(total_weights * 2)
    if variant in result.get("selections", {}):
        return int(low_payload + selected_extra_bytes(result["selections"][variant]))
    return variant_payload_bytes(result, variant, low_payload, total_weights)


def normalize_variant(raw: str, args) -> tuple[str, bool, str]:
    if raw == "pmra":
        return args.pmra_variant, False, "pmra"
    if raw == "pmra_orbitquant":
        return args.pmra_variant, True, "pmra_orbitquant"
    if raw == "orbitquant":
        return args.orbit_base_source, True, "orbitquant"
    suffix = "_orbitquant"
    if raw.endswith(suffix):
        return raw[: -len(suffix)], True, raw
    return raw, False, raw


def patch_static_variant(
    model,
    hf,
    readers: dict[str, dict],
    groups: dict,
    specs,
    result: dict,
    base_variant: str,
    tensor_profile: str,
) -> None:
    layers = result["args"]["layers"]
    group_mode = result["args"]["group_mode"]
    low_source = result["args"]["low_source"]
    if base_variant == "fp16":
        print("[orbit-gemma4] restoring fp16 reference weights", flush=True)
        patch_all_from_hf(model, hf, specs)
    elif base_variant in result.get("selections", {}):
        print(f"[orbit-gemma4] patching PMRA selection {base_variant}", flush=True)
        patch_all_from_source(model, hf, readers, layers, low_source, group_mode, tensor_profile)
        apply_selection(model, hf, readers, groups, result["selections"][base_variant])
    else:
        print(f"[orbit-gemma4] patching uniform source {base_variant}", flush=True)
        patch_all_from_source(model, hf, readers, layers, base_variant, group_mode, tensor_profile)


def estimate_runtime_savings_mib(args, text_config: dict | object, orbit_audit: dict | None = None) -> dict[str, float]:
    def cfg(name: str, default: int) -> int:
        if isinstance(text_config, dict):
            return int(text_config.get(name, default))
        return int(getattr(text_config, name, default))

    kv_layers = len(parse_int_list(args.kv_layers))
    mlp_layers = len(parse_mlp_choices(args.mlp_choices, args.mlp_alpha))
    head_dim = cfg("head_dim", 256)
    kv_heads = cfg("num_key_value_heads", 1)
    ctx = int(args.memory_context_length)
    live = int(args.mlp_lifetime_tokens)
    kv_fp16 = kv_layers * 2 * ctx * kv_heads * head_dim * 2
    kv_quant = kv_layers * 2 * ctx * kv_heads * head_dim * args.kv_bits / 8
    if orbit_audit:
        mlp_dims = [
            int(item.get("rotation_summary", {}).get("dim", cfg("intermediate_size", 6144)))
            for item in orbit_audit.get("mlp_choices", [])
        ]
    else:
        mlp_dims = [cfg("intermediate_size", 6144)] * mlp_layers
    mlp_elements = live * sum(mlp_dims)
    mlp_fp16 = mlp_elements * 2
    mlp_quant = mlp_elements * args.mlp_bits / 8
    return {
        "context_length": ctx,
        "mlp_lifetime_tokens": live,
        "kv_saved_mib": float((kv_fp16 - kv_quant) / (1024 * 1024)),
        "mlp_saved_mib": float((mlp_fp16 - mlp_quant) / (1024 * 1024)),
        "total_saved_mib": float((kv_fp16 + mlp_fp16 - kv_quant - mlp_quant) / (1024 * 1024)),
        "mlp_dims": mlp_dims,
    }


def make_markdown(payload: dict) -> str:
    mlp_choices = ", ".join(
        f"{item['layer']}:{item['primitive']}:{item['rotation']}"
        for item in payload["orbit_policy"]["mlp_choices"]
    )
    lines = [
        "# Result Card - Gemma4 PMRA x OrbitQuant Stack",
        "",
        "## Variants",
        "",
        "| Variant | Static Base | OrbitQuant | NLL | Delta vs FP16 | Payload bpw | Payload bytes | Last-logit MSE | Top-10 overlap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["variants"].items():
        lines.append(
            f"| {name} | {row['static_base']} | {str(row['orbitquant']).lower()} | "
            f"{row['nll']:.6f} | {row['delta_nll_vs_fp16']:.6f} | {row['payload_bpw']:.6f} | "
            f"{row['payload_bytes']} | "
            f"{row.get('last_logit_mse_to_fp16', float('nan')):.6g} | "
            f"{row.get('top10_overlap_to_fp16', float('nan')):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Runtime Saving Estimate",
            "",
            f"- context length: `{payload['runtime_savings_estimate']['context_length']}`",
            f"- MLP live tokens: `{payload['runtime_savings_estimate']['mlp_lifetime_tokens']}`",
            f"- KV saved MiB: `{payload['runtime_savings_estimate']['kv_saved_mib']:.2f}`",
            f"- MLP saved MiB: `{payload['runtime_savings_estimate']['mlp_saved_mib']:.2f}`",
            f"- total saved MiB: `{payload['runtime_savings_estimate']['total_saved_mib']:.2f}`",
            "",
            "## Policy",
            "",
            f"- KV layers: `{', '.join(map(str, payload['orbit_policy']['kv_layers']))}`",
            f"- MLP choices: `{mlp_choices}`",
            "",
        ]
    )
    return "\n".join(lines)


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
        log_prefix="[orbit-gemma4]",
    )
    groups = group_specs(specs)

    eval_prompts, eval_audit = load_public_prompts(
        tokenizer,
        args.dataset,
        None if args.dataset_config.lower() in {"", "none", "null"} else args.dataset_config,
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
        None if args.dataset_config.lower() in {"", "none", "null"} else args.dataset_config,
        args.calib_split,
        args.text_column,
        args.calib_prompt_count,
        args.calib_prompt_seed,
        args.calib_max_length,
        args.min_tokens,
    )

    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    output: dict = {
        "created_utc": datetime.now(UTC).isoformat(),
        "experiment": "gemma4_pmra_orbit_stack_eval",
        "args": {
            "model_dir": str(args.model_dir),
            "hf": str(args.hf),
            "result_json": str(args.result_json),
            "variants": variants,
            "pmra_variant": args.pmra_variant,
            "orbit_base_source": args.orbit_base_source,
            "tensor_profile": tensor_profile,
            "device": args.device,
        },
        "eval_prompt_audit": eval_audit,
        "calibration_prompt_audit": calib_audit,
        "skipped_model_tensors": [spec.logical_name for spec in skipped_specs],
        "variants": {},
        "orbit_policy": {
            "kv_layers": parse_int_list(args.kv_layers),
            "kv_bits": args.kv_bits,
            "kv_rotation": args.kv_rotation,
            "kv_alpha": args.kv_alpha,
            "mlp_bits": args.mlp_bits,
            "mlp_block_size": args.mlp_block_size,
            "mlp_choices": parse_mlp_choices(args.mlp_choices, args.mlp_alpha),
        },
    }

    with open_hf_tensor_source(args.hf) as hf:
        total_weights = total_weight_count(hf, model, specs)
        low_payload = source_payload_from_result(result, result["args"]["low_source"])
        fp_last_logits = None
        fp_nll = None

        for raw_variant in variants:
            base_variant, use_orbit, display_name = normalize_variant(raw_variant, args)
            print(
                f"[orbit-gemma4] evaluating {display_name} base={base_variant} orbit={use_orbit}",
                flush=True,
            )
            patch_static_variant(model, hf, readers, groups, specs, result, base_variant, tensor_profile)

            handles = []
            orbit_audit = None
            try:
                if use_orbit:
                    handles, orbit_audit = apply_orbitquant(model, tokenizer, calib_prompts, args)

                eval_args = SimpleNamespace(eval_max_length=args.eval_max_length)
                eval_result = evaluate_model(
                    model,
                    tokenizer,
                    eval_prompts,
                    eval_args,
                    fp_last_logits,
                )
                if display_name == "fp16":
                    fp_last_logits = eval_result["captured_last_logits"]
                    fp_nll = float(eval_result["nll"])
                    stripped = strip_logits(eval_result)
                else:
                    stripped = strip_logits(eval_result)
                if fp_nll is None:
                    fp_nll = float(stripped["nll"])
                payload_bytes = static_payload_bytes(result, base_variant, low_payload, total_weights)
                stripped["payload_bytes"] = int(payload_bytes)
                stripped["payload_bpw"] = float(payload_bytes * 8 / total_weights)
                stripped["delta_nll_vs_fp16"] = float(stripped["nll"] - fp_nll)
                stripped["static_base"] = base_variant
                stripped["orbitquant"] = bool(use_orbit)
                if orbit_audit is not None:
                    stripped["orbit_audit"] = orbit_audit
                output["variants"][display_name] = stripped
                (args.output_dir / "partial_result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
            finally:
                restore_forwards(handles)

    text_config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    orbit_audit = None
    for row in output["variants"].values():
        if row.get("orbit_audit"):
            orbit_audit = row["orbit_audit"]
            break
    output["runtime_savings_estimate"] = estimate_runtime_savings_mib(args, text_config, orbit_audit)
    output["completed_utc"] = datetime.now(UTC).isoformat()
    return output


def main() -> int:
    default_kv_layers = ",".join(map(str, SMOL135_14BUS_TO_GEMMA4["kv_layers"]))
    default_mlp_choices = ",".join(SMOL135_14BUS_TO_GEMMA4["mlp_choices"])

    parser = argparse.ArgumentParser(description="Evaluate Gemma4 PMRA and OrbitQuant stacking.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], help="GGUF source as label=path.")
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default="fp16,q3_k_s,orbitquant,pmra,pmra_orbitquant")
    parser.add_argument("--pmra-variant", default="c2_calib_knapsack_mixed")
    parser.add_argument("--orbit-base-source", default="q3_k_s")
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
    parser.add_argument("--memory-context-length", type=int, default=8192)
    parser.add_argument("--mlp-lifetime-tokens", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(
            json.dumps(
                {
                    "kv_layers": parse_int_list(args.kv_layers),
                    "mlp_choices": parse_mlp_choices(args.mlp_choices, args.mlp_alpha),
                    "variants": [part.strip() for part in args.variants.split(",") if part.strip()],
                },
                indent=2,
            )
        )
        return 0

    start = time.time()
    payload = run(args)
    payload["wall_time_seconds"] = float(time.time() - start)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "result.md").write_text(make_markdown(payload), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "variants": payload["variants"]}, indent=2), flush=True)
    print(f"[orbit-gemma4] wrote {args.output_dir / 'result.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
