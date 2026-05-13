import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import DEFAULT_REPO, evaluate_model, tokenize
from smol135_fused_choice_allocator import (
    build_mlp_param_maps,
    explicit_kv_items,
    load_json,
    make_kv_specs,
    parse_mlp_choices,
    pick_mlp_candidate_layers,
    ranked_kv_layers,
    ranked_mlp_choices,
    run_stack,
    state_key,
    unique_best_mlp_choices,
)
from smol135_joint_allocator_probe import memory_estimate, stack_scores
from smol135_mlp_hadamard_plus_probe import load_prompts
from smol135_sweep import EVAL_PROMPTS, parse_int_list


class EvaluationBudgetExceeded(Exception):
    pass


def metric_score(metrics: dict, memory: dict, objective: str) -> float:
    scores = stack_scores(metrics, memory)
    return scores[objective]


def selected_kv_layers(state: dict) -> set[int]:
    return {item["layer"] for item in state["kv_specs"]}


def selected_mlp_layers(state: dict) -> set[int]:
    return {item["layer"] for item in state["mlp_params"]}


def state_shape(state: dict) -> tuple[int, int]:
    return len(state["kv_specs"]), len(state["mlp_params"])


def state_total_buses(state: dict) -> int:
    kv_count, mlp_count = state_shape(state)
    return kv_count + mlp_count


def summarize_state(state: dict) -> dict:
    return {
        "kv_layers": [item["layer"] for item in state["kv_specs"]],
        "mlp_choices": [
            {
                "layer": item["layer"],
                "primitive": item["primitive"],
                "rotation": item["rotation"],
                "alpha": item["alpha"],
            }
            for item in state["mlp_params"]
        ],
        "shape": {"kv": len(state["kv_specs"]), "mlp": len(state["mlp_params"])},
        "metrics": state["metrics"],
        "memory": state["memory"],
        "score": state["score"],
        "path": state["path"],
    }


def evaluate_state(
    model,
    eval_batch,
    baseline_logits,
    baseline,
    args,
    cache: dict,
    kv_specs: list[dict],
    mlp_params: list[dict],
) -> dict:
    key = state_key(kv_specs, mlp_params)
    if key not in cache:
        if args.max_evals > 0 and len(cache) >= args.max_evals:
            raise EvaluationBudgetExceeded(f"reached max evals: {args.max_evals}")
        metrics = run_stack(model, eval_batch, baseline_logits, baseline, args, kv_specs, mlp_params)
        memory = memory_estimate(model.config, kv_specs, mlp_params, args)
        cache[key] = {
            "metrics": metrics,
            "memory": memory,
            "score": metric_score(metrics, memory, args.objective),
        }
    return cache[key]


def make_state(
    model,
    eval_batch,
    baseline_logits,
    baseline,
    args,
    cache: dict,
    kv_specs: list[dict],
    mlp_params: list[dict],
    path: list[dict],
) -> dict:
    evaluated = evaluate_state(model, eval_batch, baseline_logits, baseline, args, cache, kv_specs, mlp_params)
    return {
        "kv_specs": kv_specs,
        "mlp_params": mlp_params,
        "metrics": evaluated["metrics"],
        "memory": evaluated["memory"],
        "score": evaluated["score"],
        "path": path,
    }


def alternate_primitive(primitive: str) -> str:
    if primitive == "plain":
        return "plus"
    if primitive == "plus":
        return "plain"
    raise ValueError(f"Unknown MLP primitive: {primitive}")


def refine_swaps(
    model,
    eval_batch,
    baseline_logits,
    baseline,
    args,
    cache: dict,
    state: dict,
    mlp_param_map: dict[tuple[int, str], dict],
) -> dict:
    current = state
    for _ in range(args.swap_rounds):
        candidates = []
        for item in current["mlp_params"]:
            layer = item["layer"]
            primitive = alternate_primitive(item["primitive"])
            trial_mlp = [
                mlp_param_map[(param["layer"], primitive)] if param["layer"] == layer else param
                for param in current["mlp_params"]
            ]
            trial = make_state(
                model,
                eval_batch,
                baseline_logits,
                baseline,
                args,
                cache,
                current["kv_specs"],
                trial_mlp,
                current["path"]
                + [
                    {
                        "op": "swap_mlp",
                        "layer": layer,
                        "primitive": primitive,
                    }
                ],
            )
            candidates.append(trial)

        if not candidates:
            break
        best = min(candidates, key=lambda item: (item["score"], item["metrics"]["kl_from_baseline"]))
        if best["score"] < current["score"] - args.min_swap_improvement:
            current = best
            print(
                f"refine swap layer={current['path'][-1]['layer']} "
                f"primitive={current['path'][-1]['primitive']} "
                f"kl={current['metrics']['kl_from_baseline']:.6f} score={current['score']:.6f}"
            )
        else:
            break
    return current


def sort_key(state: dict) -> tuple[float, float, float]:
    return (
        state["score"],
        state["metrics"]["kl_from_baseline"],
        -state["memory"]["total_saved_mb"],
    )


def prune_by_shape(states: list[dict], per_shape: int, global_cap: int) -> list[dict]:
    grouped = defaultdict(list)
    for state in states:
        grouped[state_shape(state)].append(state)

    kept = []
    for group in grouped.values():
        group = sorted(group, key=sort_key)
        kept.extend(group[:per_shape])

    kept = sorted(kept, key=sort_key)
    if global_cap > 0:
        kept = kept[:global_cap]
    return kept


def expand_state(
    model,
    eval_batch,
    baseline_logits,
    baseline,
    args,
    cache: dict,
    state: dict,
    kv_candidates: list[dict],
    mlp_candidate_layers: list[int],
    mlp_param_map: dict[tuple[int, str], dict],
) -> list[dict]:
    expanded = []
    used_kv = selected_kv_layers(state)
    used_mlp = selected_mlp_layers(state)

    for item in kv_candidates:
        if item["layer"] in used_kv:
            continue
        kv_specs = state["kv_specs"] + [{"layer": item["layer"], "rotation": item["rotation"]}]
        candidate = make_state(
            model,
            eval_batch,
            baseline_logits,
            baseline,
            args,
            cache,
            kv_specs,
            state["mlp_params"],
            state["path"]
            + [
                {
                    "op": "add_kv",
                    "layer": item["layer"],
                    "rotation": item["rotation"],
                }
            ],
        )
        expanded.append(refine_swaps(model, eval_batch, baseline_logits, baseline, args, cache, candidate, mlp_param_map))

    for layer in mlp_candidate_layers:
        if layer in used_mlp:
            continue
        for primitive in ["plain", "plus"]:
            mlp_params = state["mlp_params"] + [mlp_param_map[(layer, primitive)]]
            candidate = make_state(
                model,
                eval_batch,
                baseline_logits,
                baseline,
                args,
                cache,
                state["kv_specs"],
                mlp_params,
                state["path"]
                + [
                    {
                        "op": "add_mlp",
                        "layer": layer,
                        "primitive": primitive,
                    }
                ],
            )
            expanded.append(refine_swaps(model, eval_batch, baseline_logits, baseline, args, cache, candidate, mlp_param_map))

    return expanded


def run_search(args) -> dict:
    kv_payload = load_json(args.kv_impact_result)
    local_payload = load_json(args.local_result)
    mlp_choice_payload = load_json(args.mlp_choice_impact_result)
    kv_ranking = ranked_kv_layers(kv_payload)
    kv_by_layer = {item["layer"]: item for item in kv_ranking}
    mlp_choice_ranking = ranked_mlp_choices(mlp_choice_payload)

    if args.base_kv_layers:
        base_kv_items = explicit_kv_items(parse_int_list(args.base_kv_layers), kv_by_layer)
    else:
        base_kv_items = kv_ranking[: args.base_kv_count]

    if args.base_mlp_choices:
        base_mlp_choices = parse_mlp_choices(args.base_mlp_choices)
    else:
        base_mlp_choices = unique_best_mlp_choices(mlp_choice_ranking)[: args.base_mlp_count]

    base_kv_layers = {item["layer"] for item in base_kv_items}
    base_mlp_layers = {item["layer"] for item in base_mlp_choices}
    kv_candidates = [item for item in kv_ranking if item["layer"] not in base_kv_layers][: args.kv_candidate_count]
    mlp_candidate_layers = pick_mlp_candidate_layers(
        mlp_choice_ranking,
        base_mlp_layers,
        args.mlp_candidate_count,
    )
    mlp_layers_to_build = sorted(base_mlp_layers | set(mlp_candidate_layers))

    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.float32)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    model.eval()

    calibration_prompts = load_prompts(args.calibration_prompts, [])
    eval_prompts = load_prompts(args.eval_prompts, EVAL_PROMPTS)
    if args.calibration_prompt_limit > 0:
        calibration_prompts = calibration_prompts[: args.calibration_prompt_limit]
    if args.eval_prompt_limit > 0:
        eval_prompts = eval_prompts[: args.eval_prompt_limit]
    cal_batch = tokenize(tokenizer, calibration_prompts, args.max_length)
    eval_batch = tokenize(tokenizer, eval_prompts, args.max_length)
    cal_batch = {key: value.to(device) for key, value in cal_batch.items()}
    eval_batch = {key: value.to(device) for key, value in eval_batch.items()}
    baseline = evaluate_model(model, eval_batch, topk=args.topk)
    baseline_logits = baseline.pop("logits")

    mlp_param_map = build_mlp_param_maps(args, model, cal_batch, local_payload, mlp_layers_to_build)
    base_kv_specs = make_kv_specs(base_kv_items)
    base_mlp_params = [mlp_param_map[(item["layer"], item["primitive"])] for item in base_mlp_choices]
    cache = {}

    base_state = make_state(
        model,
        eval_batch,
        baseline_logits,
        baseline,
        args,
        cache,
        base_kv_specs,
        base_mlp_params,
        [],
    )
    base_state = refine_swaps(model, eval_batch, baseline_logits, baseline, args, cache, base_state, mlp_param_map)
    beam = [base_state]
    frontiers = [
        {
            "total_buses": state_total_buses(base_state),
            "best": summarize_state(base_state),
            "beam": [summarize_state(base_state)],
        }
    ]
    print(
        f"policy base buses={state_total_buses(base_state)} "
        f"kl={base_state['metrics']['kl_from_baseline']:.6f} "
        f"score={base_state['score']:.6f}"
    )

    stopped_reason = None
    for step in range(args.max_extra_buses):
        candidates = {}
        try:
            for state in beam:
                for expanded in expand_state(
                    model,
                    eval_batch,
                    baseline_logits,
                    baseline,
                    args,
                    cache,
                    state,
                    kv_candidates,
                    mlp_candidate_layers,
                    mlp_param_map,
                ):
                    key = state_key(expanded["kv_specs"], expanded["mlp_params"])
                    old = candidates.get(key)
                    if old is None or sort_key(expanded) < sort_key(old):
                        candidates[key] = expanded
        except EvaluationBudgetExceeded as exc:
            stopped_reason = str(exc)
            print(f"policy stop {stopped_reason}")
            break

        if not candidates:
            break

        beam = prune_by_shape(
            list(candidates.values()),
            args.beam_width_per_shape,
            args.global_beam_cap,
        )
        best = min(beam, key=sort_key)
        total_buses = state_total_buses(best)
        frontiers.append(
            {
                "total_buses": total_buses,
                "best": summarize_state(best),
                "beam": [summarize_state(state) for state in beam],
            }
        )
        print(
            f"policy step={step + 1} buses={total_buses} "
            f"kl={best['metrics']['kl_from_baseline']:.6f} "
            f"topk={best['metrics']['topk_overlap']:.6f} "
            f"score={best['score']:.6f} states={len(beam)} evals={len(cache)}"
        )

    return {
        "experiment": "fused_policy_search",
        "repo": args.repo,
        "objective": args.objective,
        "kv_impact_result": str(args.kv_impact_result),
        "local_result": str(args.local_result),
        "mlp_choice_impact_result": str(args.mlp_choice_impact_result),
        "baseline_eval": baseline,
        "base_kv_items": base_kv_items,
        "base_mlp_choices": base_mlp_choices,
        "kv_candidates": kv_candidates,
        "mlp_candidate_layers": mlp_candidate_layers,
        "frontiers": frontiers,
        "stopped_reason": stopped_reason,
        "summary": {
            "frontier": [
                {
                    "total_buses": item["total_buses"],
                    "kv_layers": item["best"]["kv_layers"],
                    "mlp_choices": item["best"]["mlp_choices"],
                    "kl": item["best"]["metrics"]["kl_from_baseline"],
                    "topk": item["best"]["metrics"]["topk_overlap"],
                    "loss_delta": item["best"]["metrics"]["loss_delta_from_baseline"],
                    "saved_mb": item["best"]["memory"]["total_saved_mb"],
                    "score": item["best"]["score"],
                }
                for item in frontiers
            ],
            "evaluated_states": len(cache),
            "stopped_reason": stopped_reason,
        },
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_fused_policy_search_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fused policy search over KV sites and MLP primitive choices")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--kv-impact-result", type=Path, required=True)
    parser.add_argument("--local-result", type=Path, required=True)
    parser.add_argument("--mlp-choice-impact-result", type=Path, required=True)
    parser.add_argument("--base-kv-count", type=int, default=4)
    parser.add_argument("--base-mlp-count", type=int, default=4)
    parser.add_argument("--base-kv-layers")
    parser.add_argument("--base-mlp-choices")
    parser.add_argument("--kv-candidate-count", type=int, default=8)
    parser.add_argument("--mlp-candidate-count", type=int, default=8)
    parser.add_argument("--max-extra-buses", type=int, default=8)
    parser.add_argument("--beam-width-per-shape", type=int, default=2)
    parser.add_argument("--global-beam-cap", type=int, default=24)
    parser.add_argument("--swap-rounds", type=int, default=2)
    parser.add_argument("--min-swap-improvement", type=float, default=0.0)
    parser.add_argument("--objective", choices=["kl", "kl_per_saved_mb"], default="kl")
    parser.add_argument("--kv-bits", type=int, default=3)
    parser.add_argument("--kv-alpha", type=float, default=0.75)
    parser.add_argument("--mlp-bits", type=int, default=2)
    parser.add_argument("--full-bits", type=int, default=16)
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--memory-context-length", type=int, default=2048)
    parser.add_argument("--mlp-lifetime-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--calibration-prompts", required=True)
    parser.add_argument("--eval-prompts")
    parser.add_argument("--calibration-prompt-limit", type=int, default=0)
    parser.add_argument("--eval-prompt-limit", type=int, default=0)
    parser.add_argument("--max-evals", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    result = run_search(args)
    path = save_result(result, args.out_dir)
    print(json.dumps({"experiment": result["experiment"], "summary": result["summary"]}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
