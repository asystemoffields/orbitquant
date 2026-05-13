import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import (
    DEFAULT_REPO,
    get_attention_shape,
    get_layer,
    hadamard,
    random_orthogonal,
    tokenize,
)
from smol135_sweep import EVAL_PROMPTS, parse_int_list, parse_str_list


def capture_qkv(model, batch, layer_idx: int) -> dict[str, torch.Tensor]:
    layer = get_layer(model, layer_idx)
    captured = {}
    handles = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            captured[name] = output.detach().float().cpu()

        return hook

    handles.append(layer.self_attn.q_proj.register_forward_hook(make_hook("q")))
    handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook("k")))
    handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook("v")))
    with torch.no_grad():
        model(**batch)
    for handle in handles:
        handle.remove()
    return captured


def rotation_matrix(dim: int, rotation: str, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if rotation == "identity":
        return torch.eye(dim)
    if rotation == "hadamard":
        return hadamard(dim)
    if rotation == "random_orthogonal":
        return random_orthogonal(dim, generator)
    raise ValueError(f"Unknown rotation: {rotation}")


def quantize_vectors(vectors: torch.Tensor, bits: int, rotation: torch.Tensor) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    rotated = vectors @ rotation
    scale = rotated.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    quantized = torch.round(rotated / scale).clamp(-qmax, qmax) * scale
    return quantized @ rotation.T


def rel_mse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((((reference - candidate) ** 2).mean() / (reference**2).mean().clamp_min(1e-12)).item())


def reshape_heads(tensor: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], tensor.shape[1], num_heads, head_dim)


def key_score_rel_mse(
    q_heads: torch.Tensor,
    k_head: torch.Tensor,
    k_quant: torch.Tensor,
) -> float:
    errors = []
    for batch_idx in range(q_heads.shape[0]):
        keys = k_head[batch_idx]
        keys_q = k_quant[batch_idx]
        for head_idx in range(q_heads.shape[2]):
            query = q_heads[batch_idx, :, head_idx, :]
            ref = query @ keys.T / math.sqrt(query.shape[-1])
            cand = query @ keys_q.T / math.sqrt(query.shape[-1])
            errors.append(rel_mse(ref, cand))
    return float(sum(errors) / len(errors))


def run_probe(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.float32)
    model.eval()
    batch = tokenize(tokenizer, EVAL_PROMPTS, args.max_length)

    _, num_heads, num_kv_heads, head_dim = get_attention_shape(model)
    heads_per_kv = num_heads // num_kv_heads
    layers = parse_int_list(args.layers)
    rotations = parse_str_list(args.rotations)
    rows = []

    for layer_idx in layers:
        captured = capture_qkv(model, batch, layer_idx)
        q = reshape_heads(captured["q"], num_heads, head_dim)
        k = reshape_heads(captured["k"], num_kv_heads, head_dim)
        v = reshape_heads(captured["v"], num_kv_heads, head_dim)

        for kv_head in range(num_kv_heads):
            q_start = kv_head * heads_per_kv
            q_stop = q_start + heads_per_kv
            q_group = q[:, :, q_start:q_stop, :]
            k_head = k[:, :, kv_head, :]
            v_head = v[:, :, kv_head, :]

            for rotation_name in rotations:
                rot = rotation_matrix(
                    head_dim,
                    rotation_name,
                    args.seed + layer_idx * 1009 + kv_head * 97 + len(rotation_name),
                )
                k_quant = quantize_vectors(k_head, args.bits, rot)
                v_quant = quantize_vectors(v_head, args.bits, rot)
                row = {
                    "layer": layer_idx,
                    "kv_head": kv_head,
                    "rotation": rotation_name,
                    "bits": args.bits,
                    "key_vector_rel_mse": rel_mse(k_head, k_quant),
                    "value_vector_rel_mse": rel_mse(v_head, v_quant),
                    "key_attention_score_rel_mse": key_score_rel_mse(q_group, k_head, k_quant),
                }
                rows.append(row)
            best_key = min(
                [row for row in rows if row["layer"] == layer_idx and row["kv_head"] == kv_head],
                key=lambda row: row["key_attention_score_rel_mse"],
            )
            print(
                f"kv layer={layer_idx} head={kv_head} "
                f"best_score={best_key['rotation']} "
                f"score_mse={best_key['key_attention_score_rel_mse']:.6f}"
            )

    grouped = {}
    for row in rows:
        key = (row["layer"], row["kv_head"])
        grouped.setdefault(key, []).append(row)

    wins = {rotation: 0 for rotation in rotations}
    key_score_deltas = []
    key_vector_deltas = []
    value_vector_deltas = []
    for group_rows in grouped.values():
        identity = next(row for row in group_rows if row["rotation"] == "identity")
        best = min(group_rows, key=lambda row: row["key_attention_score_rel_mse"])
        wins[best["rotation"]] += 1
        key_score_deltas.append(best["key_attention_score_rel_mse"] - identity["key_attention_score_rel_mse"])
        key_vector_deltas.append(best["key_vector_rel_mse"] - identity["key_vector_rel_mse"])
        value_vector_deltas.append(best["value_vector_rel_mse"] - identity["value_vector_rel_mse"])

    return {
        "experiment": "kv_rotation_probe",
        "repo": args.repo,
        "layers": layers,
        "bits": args.bits,
        "rotations": rotations,
        "eval_prompt_count": len(EVAL_PROMPTS),
        "summary": {
            "groups": len(grouped),
            "best_rotation_counts": wins,
            "mean_best_key_score_delta_vs_identity": sum(key_score_deltas) / len(key_score_deltas),
            "mean_best_key_vector_delta_vs_identity": sum(key_vector_deltas) / len(key_vector_deltas),
            "mean_best_value_vector_delta_vs_identity": sum(value_vector_deltas) / len(value_vector_deltas),
        },
        "rows": rows,
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_kv_rotation_{payload['bits']}bit_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant-style rotation probe on Smol135 KV activations")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--layers", default="0,1,2,4,8,12,16,20,24,29")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--rotations", default="identity,hadamard,random_orthogonal")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    result = run_probe(args)
    path = save_result(result, args.out_dir)
    print(json.dumps({"experiment": result["experiment"], "summary": result["summary"]}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
