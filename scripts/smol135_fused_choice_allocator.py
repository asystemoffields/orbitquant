import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import DEFAULT_REPO, evaluate_model, tokenize
from smol135_fused_hadamard_plus_probe import run_candidate
from smol135_fused_stack_probe import patch_attention_layers
from smol135_mlp_hadamard_plus_impact_sweep import build_layer_params, load_json, patch_mlp_precomputed
from smol135_mlp_hadamard_plus_probe import load_prompts
from smol135_mlp_stack_probe import restore_forwards, summarize_candidate
from smol135_sweep import EVAL_PROMPTS, parse_int_list


def ranked_kv_layers(payload: dict) -> list[dict]:
    rows = []
    for row in payload["rows"]:
        best = row["best_by_calibration"]
        metrics = row["candidates"][best]["eval"]
        rows.append(
            {
                "kind": "kv",
                "layer": row["layer"],
                "rotation": best,
                "independent_kl": metrics["kl_from_baseline"],
            }
        )
    return sorted(rows, key=lambda item: item["independent_kl"])


def ranked_mlp_choices(payload: dict) -> list[dict]:
    choices = []
    for row in payload["rows"]:
        layer = row["layer"]
        for key, primitive in [
            ("plain_block_hadamard", "plain"),
            ("hadamard_plus", "plus"),
        ]:
            choices.append(
                {
                    "kind": "mlp",
                    "layer": layer,
                    "primitive": primitive,
                    "independent_kl": row[key]["metrics"]["kl_from_baseline"],
                }
            )
    return sorted(choices, key=lambda item: item["independent_kl"])


def unique_best_mlp_choices(choices: list[dict]) -> list[dict]:
    seen = set()
    selected = []
    for item in choices:
        if item["layer"] in seen:
            continue
        seen.add(item["layer"])
        selected.append(item)
    return selected


def pick_base_mlp(choices: list[dict], count: int) -> list[dict]:
    return unique_best_mlp_choices(choices)[:count]


def pick_mlp_candidate_layers(choices: list[dict], excluded_layers: set[int], count: int) -> list[int]:
    layers = []
    for item in unique_best_mlp_choices(choices):
        if item["layer"] in excluded_layers:
            continue
        layers.append(item["layer"])
        if len(layers) >= count:
            break
    return layers


def make_kv_specs(items: list[dict]) -> list[dict]:
    return [{"layer": item["layer"], "rotation": item["rotation"]} for item in items]


def parse_mlp_choices(spec: str) -> list[dict]:
    choices = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("MLP choices must use layer:primitive, for example 16:plain")
        layer_text, primitive = part.split(":", 1)
        primitive = primitive.strip()
        if primitive not in {"plain", "plus"}:
            raise ValueError("MLP primitive must be plain or plus")
        choices.append(
            {
                "kind": "mlp",
                "layer": int(layer_text.strip()),
                "primitive": primitive,
                "independent_kl": 0.0,
            }
        )
    return choices


def explicit_kv_items(layers: list[int], kv_by_layer: dict[int, dict]) -> list[dict]:
    return [kv_by_layer[layer] for layer in layers]


def patch_kv_specs(model, specs: list[dict], bits: int, alpha: float, seed: int) -> list:
    saved = []
    for rotation in sorted({item["rotation"] for item in specs}):
        layers = [item["layer"] for item in specs if item["rotation"] == rotation]
        saved += patch_attention_layers(model, layers, bits, rotation, alpha, seed)
    return saved


def run_stack(
    model,
    eval_batch,
    baseline_logits,
    baseline,
    args,
    kv_specs: list[dict],
    mlp_params: list[dict],
) -> dict:
    saved_kv = patch_kv_specs(model, kv_specs, args.kv_bits, args.kv_alpha, args.seed) if kv_specs else []
    saved_mlp = patch_mlp_precomputed(model, mlp_params, args.mlp_bits) if mlp_params else []
    metrics = summarize_candidate(model, eval_batch, baseline_logits, baseline, args.topk)
    restore_forwards(saved_mlp)
    restore_forwards(saved_kv)
    return metrics


def state_key(kv_specs: list[dict], mlp_params: list[dict]) -> tuple:
    kv = tuple(sorted((item["layer"], item["rotation"]) for item in kv_specs))
    mlp = tuple(sorted((item["layer"], item["primitive"]) for item in mlp_params))
    return kv, mlp


def selected_kv_layers(kv_specs: list[dict]) -> set[int]:
    return {item["layer"] for item in kv_specs}


def selected_mlp_layers(mlp_params: list[dict]) -> set[int]:
    return {item["layer"] for item in mlp_params}


def summarize_state(state: dict) -> dict:
    return {
        "kv_layers": [item["layer"] for item in state["kv_specs"]],
        "mlp_choices": [
            {"layer": item["layer"], "primitive": item["primitive"], "rotation": item["rotation"], "alpha": item["alpha"]}
            for item in state["mlp_params"]
        ],
        "metrics": state["metrics"],
        "path": state["path"],
    }


def build_mlp_param_maps(args, model, cal_batch, local_result: dict, layers: list[int]) -> dict[tuple[int, str], dict]:
    local_rows = [row for row in sorted(local_result["rows"], key=lambda row: row["layer"]) if row["layer"] in set(layers)]
    plus_params = build_layer_params(args, model, cal_batch, local_rows, "best")
    plain_params = build_layer_params(args, model, cal_batch, local_rows, "plain")

    params = {}
    for item in plus_params:
        item = dict(item)
        item["primitive"] = "plus"
        params[(item["layer"], "plus")] = item
    for item in plain_params:
        item = dict(item)
        item["primitive"] = "plain"
        params[(item["layer"], "plain")] = item
    return params


def run_beam(
    model,
    eval_batch,
    baseline_logits,
    baseline,
    args,
    base_kv_specs: list[dict],
    base_mlp_params: list[dict],
    kv_candidates: list[dict],
    mlp_candidates: list[dict],
    mlp_param_map: dict[tuple[int, str], dict],
) -> dict:
    base_metrics = run_stack(model, eval_batch, baseline_logits, baseline, args, base_kv_specs, base_mlp_params)
    print(
        f"base choice stack kl={base_metrics['kl_from_baseline']:.6f} "
        f"topk={base_metrics['topk_overlap']:.6f}"
    )
    beam = [{"kv_specs": base_kv_specs, "mlp_params": base_mlp_params, "metrics": base_metrics, "path": []}]
    frontiers = []

    for step_idx in range(args.beam_steps):
        expanded = {}
        for state in beam:
            used_kv = selected_kv_layers(state["kv_specs"])
            used_mlp = selected_mlp_layers(state["mlp_params"])

            additions = []
            additions.extend([item for item in kv_candidates if item["layer"] not in used_kv])
            additions.extend([item for item in mlp_candidates if item["layer"] not in used_mlp])

            for item in additions:
                if item["kind"] == "kv":
                    kv_specs = state["kv_specs"] + [{"layer": item["layer"], "rotation": item["rotation"]}]
                    mlp_params = state["mlp_params"]
                    label = f"kv_{item['layer']}_{item['rotation']}"
                else:
                    kv_specs = state["kv_specs"]
                    mlp_params = state["mlp_params"] + [mlp_param_map[(item["layer"], item["primitive"])]]
                    label = f"mlp_{item['layer']}_{item['primitive']}"

                metrics = run_stack(model, eval_batch, baseline_logits, baseline, args, kv_specs, mlp_params)
                new_state = {
                    "kv_specs": kv_specs,
                    "mlp_params": mlp_params,
                    "metrics": metrics,
                    "path": state["path"]
                    + [
                        {
                            "add": label,
                            "kind": item["kind"],
                            "layer": item["layer"],
                            "primitive": item.get("primitive"),
                            "independent_kl": item["independent_kl"],
                            "stack_kl": metrics["kl_from_baseline"],
                            "stack_topk": metrics["topk_overlap"],
                        }
                    ],
                }
                key = state_key(kv_specs, mlp_params)
                old = expanded.get(key)
                if old is None or metrics["kl_from_baseline"] < old["metrics"]["kl_from_baseline"]:
                    expanded[key] = new_state

        ranked = sorted(expanded.values(), key=lambda state: state["metrics"]["kl_from_baseline"])
        beam = ranked[: args.beam_width]
        best = beam[0]
        last = best["path"][-1]
        frontiers.append({"added_buses": step_idx + 1, "best": summarize_state(best), "beam": [summarize_state(s) for s in beam]})
        print(
            f"beam step={step_idx + 1} best_add={last['add']} "
            f"kl={best['metrics']['kl_from_baseline']:.6f} "
            f"topk={best['metrics']['topk_overlap']:.6f}"
        )

    return {
        "base": summarize_state(beam[0]) if not frontiers else None,
        "frontiers": frontiers,
        "final_beam": [summarize_state(state) for state in beam],
        "base_metrics": base_metrics,
    }


def run_probe(args) -> dict:
    kv_payload = load_json(args.kv_impact_result)
    local_payload = load_json(args.local_result)
    mlp_choice_payload = load_json(args.mlp_choice_impact_result)

    kv_ranking = ranked_kv_layers(kv_payload)
    mlp_choice_ranking = ranked_mlp_choices(mlp_choice_payload)
    kv_by_layer = {item["layer"]: item for item in kv_ranking}

    if args.base_kv_layers:
        base_kv_items = explicit_kv_items(parse_int_list(args.base_kv_layers), kv_by_layer)
    else:
        base_kv_items = kv_ranking[: args.base_kv_count]
    if args.base_mlp_choices:
        base_mlp_choices = parse_mlp_choices(args.base_mlp_choices)
        choice_lookup = {(item["layer"], item["primitive"]): item for item in mlp_choice_ranking}
        for item in base_mlp_choices:
            item["independent_kl"] = choice_lookup[(item["layer"], item["primitive"])]["independent_kl"]
    else:
        base_mlp_choices = pick_base_mlp(mlp_choice_ranking, args.base_mlp_count)
    kv_candidates = [item for item in kv_ranking if item["layer"] not in {base["layer"] for base in base_kv_items}][
        : args.candidate_count
    ]
    mlp_candidate_layers = pick_mlp_candidate_layers(
        mlp_choice_ranking,
        {base["layer"] for base in base_mlp_choices},
        args.candidate_count,
    )
    mlp_candidates = [
        item
        for item in mlp_choice_ranking
        if item["layer"] in set(mlp_candidate_layers)
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.float32)
    model.eval()

    calibration_prompts = load_prompts(args.calibration_prompts, [])
    eval_prompts = load_prompts(args.eval_prompts, EVAL_PROMPTS)
    cal_batch = tokenize(tokenizer, calibration_prompts, args.max_length)
    eval_batch = tokenize(tokenizer, eval_prompts, args.max_length)
    baseline = evaluate_model(model, eval_batch, topk=args.topk)
    baseline_logits = baseline.pop("logits")

    mlp_layers_to_build = sorted({item["layer"] for item in base_mlp_choices} | set(mlp_candidate_layers))
    mlp_param_map = build_mlp_param_maps(args, model, cal_batch, local_payload, mlp_layers_to_build)
    base_kv_specs = make_kv_specs(base_kv_items)
    base_mlp_params = [mlp_param_map[(item["layer"], item["primitive"])] for item in base_mlp_choices]

    beam = run_beam(
        model,
        eval_batch,
        baseline_logits,
        baseline,
        args,
        base_kv_specs,
        base_mlp_params,
        kv_candidates,
        mlp_candidates,
        mlp_param_map,
    )

    return {
        "experiment": "fused_choice_allocator",
        "repo": args.repo,
        "kv_impact_result": str(args.kv_impact_result),
        "local_result": str(args.local_result),
        "mlp_choice_impact_result": str(args.mlp_choice_impact_result),
        "base_kv_items": base_kv_items,
        "base_mlp_choices": base_mlp_choices,
        "kv_candidates": kv_candidates,
        "mlp_candidates": mlp_candidates,
        "baseline_eval": baseline,
        "beam": beam,
        "summary": {
            "frontier": [
                {
                    "added_buses": row["added_buses"],
                    "kv_layers": row["best"]["kv_layers"],
                    "mlp_choices": row["best"]["mlp_choices"],
                    "kl": row["best"]["metrics"]["kl_from_baseline"],
                    "topk": row["best"]["metrics"]["topk_overlap"],
                    "loss_delta": row["best"]["metrics"]["loss_delta_from_baseline"],
                }
                for row in beam["frontiers"]
            ]
        },
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_fused_choice_allocator_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fused allocator with per-layer MLP primitive choices")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--kv-impact-result", type=Path, required=True)
    parser.add_argument("--local-result", type=Path, required=True)
    parser.add_argument("--mlp-choice-impact-result", type=Path, required=True)
    parser.add_argument("--base-kv-count", type=int, default=4)
    parser.add_argument("--base-mlp-count", type=int, default=4)
    parser.add_argument("--base-kv-layers")
    parser.add_argument("--base-mlp-choices")
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--beam-steps", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--kv-bits", type=int, default=3)
    parser.add_argument("--kv-alpha", type=float, default=0.75)
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
    print(json.dumps({"experiment": result["experiment"], "summary": result["summary"]}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
