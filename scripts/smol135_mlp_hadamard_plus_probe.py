import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import DEFAULT_REPO, get_layer, hadamard, tokenize
from smol135_mlp_activation_rotation_probe import capture_mlp_intermediate, rel_mse
from smol135_mlp_ternary_search import evaluate_candidate
from smol135_sweep import CALIBRATION_PROMPTS, EVAL_PROMPTS, parse_float_list, parse_int_list, parse_str_list


def load_prompts(path: str | None, fallback: list[str]) -> list[str]:
    if path is None:
        return fallback
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def block_diag_hadamard(dim: int, block_size: int) -> torch.Tensor:
    if dim % block_size:
        raise ValueError("dim must be divisible by block_size")
    block = hadamard(block_size)
    return torch.block_diag(*[block for _ in range(dim // block_size)])


def permutation_matrix(order: list[int]) -> torch.Tensor:
    dim = len(order)
    matrix = torch.zeros(dim, dim)
    matrix[torch.tensor(order, dtype=torch.long), torch.arange(dim)] = 1.0
    return matrix


def random_sign_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    signs = torch.randint(0, 2, (dim,), generator=generator, dtype=torch.float32)
    signs = signs.mul(2).sub(1)
    return torch.diag(signs)


def balanced_order(scores: torch.Tensor, block_size: int) -> list[int]:
    dim = int(scores.numel())
    if dim % block_size:
        raise ValueError("dim must be divisible by block_size")
    block_count = dim // block_size
    block_loads = [0.0 for _ in range(block_count)]
    block_slots = [[] for _ in range(block_count)]

    ranked = torch.argsort(scores.detach().float().cpu(), descending=True).tolist()
    for channel in ranked:
        available = [idx for idx in range(block_count) if len(block_slots[idx]) < block_size]
        target = min(available, key=lambda idx: (block_loads[idx], len(block_slots[idx]), idx))
        block_slots[target].append(channel)
        block_loads[target] += float(scores[channel])

    order = []
    for slots in block_slots:
        order.extend(slots)
    return order


def interleaved_order(scores: torch.Tensor, block_size: int) -> list[int]:
    dim = int(scores.numel())
    if dim % block_size:
        raise ValueError("dim must be divisible by block_size")
    block_count = dim // block_size
    ranked = torch.argsort(scores.detach().float().cpu(), descending=True).tolist()
    block_slots = [[] for _ in range(block_count)]
    direction = 1
    block = 0
    for channel in ranked:
        while len(block_slots[block]) >= block_size:
            block += direction
            if block < 0 or block >= block_count:
                direction *= -1
                block += direction
        block_slots[block].append(channel)
        block += direction
        if block < 0 or block >= block_count:
            direction *= -1
            block += direction
    return [channel for slots in block_slots for channel in slots]


def random_preperm_order(dim: int, generator: torch.Generator) -> list[int]:
    return torch.randperm(dim, generator=generator).tolist()


def make_rotation(
    name: str,
    cal_z: torch.Tensor,
    down_weight: torch.Tensor,
    block_size: int,
    seed: int,
) -> torch.Tensor:
    dim = cal_z.shape[-1]
    h = block_diag_hadamard(dim, block_size)
    generator = torch.Generator().manual_seed(seed)

    if name == "identity":
        return torch.eye(dim)
    if name == "block_hadamard":
        return h
    if name == "block_hadamard_sign":
        return random_sign_matrix(dim, generator) @ h

    activation_rms = cal_z.pow(2).mean(dim=0).sqrt()
    activation_max = cal_z.abs().amax(dim=0)
    down_norm = down_weight.norm(dim=0)
    boundary_rms = activation_rms * down_norm
    boundary_max = activation_max * down_norm

    if name == "preperm_activation_rms_hadamard":
        order = balanced_order(activation_rms, block_size)
    elif name == "preperm_activation_max_hadamard":
        order = balanced_order(activation_max, block_size)
    elif name == "preperm_down_norm_hadamard":
        order = balanced_order(down_norm, block_size)
    elif name == "preperm_boundary_rms_hadamard":
        order = balanced_order(boundary_rms, block_size)
    elif name == "preperm_boundary_max_hadamard":
        order = balanced_order(boundary_max, block_size)
    elif name == "interleave_boundary_rms_hadamard":
        order = interleaved_order(boundary_rms, block_size)
    elif name.startswith("random_preperm_hadamard_"):
        order = random_preperm_order(dim, generator)
    else:
        raise ValueError(f"Unknown rotation candidate: {name}")

    return permutation_matrix(order) @ h


def expand_rotations(rotation_names: list[str], random_trials: int) -> list[str]:
    expanded = []
    for name in rotation_names:
        if name == "random_preperm_hadamard":
            expanded.extend([f"random_preperm_hadamard_{idx}" for idx in range(random_trials)])
        else:
            expanded.append(name)
    return expanded


def summarize_layer(candidates: list[dict]) -> dict:
    identity = next(item for item in candidates if item["rotation"] == "identity" and abs(item["alpha"] - 1.0) < 1e-9)
    plain = [item for item in candidates if item["rotation"] == "block_hadamard"]
    plain_best = min(plain, key=lambda item: item["calibration"]["down_output_rel_mse"])
    best = min(candidates, key=lambda item: item["calibration"]["down_output_rel_mse"])
    return {
        "identity_absmax": identity,
        "plain_block_hadamard_best": plain_best,
        "best_by_calibration": best,
        "best_minus_plain_eval": {
            "activation_rel_mse": best["eval"]["activation_rel_mse"] - plain_best["eval"]["activation_rel_mse"],
            "down_output_rel_mse": best["eval"]["down_output_rel_mse"] - plain_best["eval"]["down_output_rel_mse"],
        },
        "best_minus_identity_eval": {
            "activation_rel_mse": best["eval"]["activation_rel_mse"] - identity["eval"]["activation_rel_mse"],
            "down_output_rel_mse": best["eval"]["down_output_rel_mse"] - identity["eval"]["down_output_rel_mse"],
        },
    }


def run_probe(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.float32)
    model.eval()

    calibration_prompts = load_prompts(args.calibration_prompts, CALIBRATION_PROMPTS)
    eval_prompts = load_prompts(args.eval_prompts, EVAL_PROMPTS)
    cal_batch = tokenize(tokenizer, calibration_prompts, args.max_length)
    eval_batch = tokenize(tokenizer, eval_prompts, args.max_length)

    layers = parse_int_list(args.layers)
    rotations = expand_rotations(parse_str_list(args.rotations), args.random_preperm_trials)
    alphas = parse_float_list(args.alphas)
    rows = []

    for layer_idx in layers:
        layer = get_layer(model, layer_idx)
        down_weight = layer.mlp.down_proj.weight.detach().float().cpu()
        cal_z, cal_y = capture_mlp_intermediate(model, cal_batch, layer_idx)
        eval_z, eval_y = capture_mlp_intermediate(model, eval_batch, layer_idx)

        candidates = []
        for rotation_name in rotations:
            rotation = make_rotation(
                rotation_name,
                cal_z,
                down_weight,
                args.block_size,
                args.seed + layer_idx * 1009 + len(rotation_name),
            )
            for alpha in alphas:
                candidates.append(
                    {
                        "rotation": rotation_name,
                        "alpha": alpha,
                        "calibration": evaluate_candidate(cal_z, cal_y, down_weight, args.bits, rotation, alpha),
                        "eval": evaluate_candidate(eval_z, eval_y, down_weight, args.bits, rotation, alpha),
                    }
                )

        summary = summarize_layer(candidates)
        rows.append({"layer": layer_idx, "summary": summary, "candidates": candidates})
        best = summary["best_by_calibration"]
        plain = summary["plain_block_hadamard_best"]
        print(
            f"hadamard-plus layer={layer_idx} "
            f"plain={plain['eval']['down_output_rel_mse']:.6f} "
            f"best={best['rotation']} alpha={best['alpha']:.3f} "
            f"best_eval={best['eval']['down_output_rel_mse']:.6f}"
        )

    wins_vs_plain = sum(1 for row in rows if row["summary"]["best_minus_plain_eval"]["down_output_rel_mse"] < 0)
    wins_vs_identity = sum(1 for row in rows if row["summary"]["best_minus_identity_eval"]["down_output_rel_mse"] < 0)
    best_counts = {}
    for row in rows:
        best = row["summary"]["best_by_calibration"]
        key = f"{best['rotation']}@{best['alpha']}"
        best_counts[key] = best_counts.get(key, 0) + 1

    return {
        "experiment": "mlp_hadamard_plus_probe",
        "repo": args.repo,
        "layers": layers,
        "bits": args.bits,
        "rotations": rotations,
        "alphas": alphas,
        "block_size": args.block_size,
        "calibration_prompt_count": len(calibration_prompts),
        "eval_prompt_count": len(eval_prompts),
        "summary": {
            "groups": len(rows),
            "wins_vs_plain_block_hadamard": wins_vs_plain,
            "wins_vs_identity_absmax": wins_vs_identity,
            "best_counts": best_counts,
            "mean_eval_down_output_delta_vs_plain": sum(
                row["summary"]["best_minus_plain_eval"]["down_output_rel_mse"] for row in rows
            )
            / len(rows),
            "mean_eval_activation_delta_vs_plain": sum(
                row["summary"]["best_minus_plain_eval"]["activation_rel_mse"] for row in rows
            )
            / len(rows),
        },
        "rows": rows,
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_mlp_hadamard_plus_{payload['bits']}bit_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Boundary-aware Hadamard-plus candidates for the 2-bit MLP bus")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--layers", default="0,1,2,4,8,12,16,20,24,29")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument(
        "--rotations",
        default=(
            "identity,block_hadamard,block_hadamard_sign,"
            "preperm_activation_rms_hadamard,preperm_activation_max_hadamard,"
            "preperm_down_norm_hadamard,preperm_boundary_rms_hadamard,"
            "preperm_boundary_max_hadamard,interleave_boundary_rms_hadamard,"
            "random_preperm_hadamard"
        ),
    )
    parser.add_argument("--random-preperm-trials", type=int, default=4)
    parser.add_argument("--alphas", default="1.0,0.75,0.5,0.375,0.25")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--calibration-prompts")
    parser.add_argument("--eval-prompts")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    result = run_probe(args)
    path = save_result(result, args.out_dir)
    print(json.dumps({"experiment": result["experiment"], "summary": result["summary"]}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
