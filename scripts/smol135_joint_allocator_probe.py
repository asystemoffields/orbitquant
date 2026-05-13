import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import DEFAULT_REPO, evaluate_model, tokenize
from smol135_fused_stack_probe import patch_attention_layers
from smol135_mlp_stack_probe import patch_mlp_layers, restore_forwards, summarize_candidate
from smol135_sweep import EVAL_PROMPTS, parse_int_list


def load_prompts(path: str | None, fallback: list[str]) -> list[str]:
    if path is None:
        return fallback
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ranked_kv_layers(payload: dict) -> list[dict]:
    rows = []
    for row in payload["rows"]:
        best = row["best_by_calibration"]
        metrics = row["candidates"][best]["eval"]
        rows.append(
            {
                "layer": row["layer"],
                "rotation": best,
                "eval_kl": metrics["kl_from_baseline"],
                "eval_topk": metrics["topk_overlap"],
                "eval_loss_delta": metrics["loss_delta_from_baseline"],
            }
        )
    return sorted(rows, key=lambda item: item["eval_kl"])


def ranked_mlp_layers(payload: dict) -> list[dict]:
    rows = []
    for row in payload["rows"]:
        metrics = row["calibrated"]
        item = dict(row["calibrated_param"])
        item.update(
            {
                "eval_kl": metrics["kl_from_baseline"],
                "eval_topk": metrics["topk_overlap"],
                "eval_loss_delta": metrics["loss_delta_from_baseline"],
            }
        )
        rows.append(item)
    return sorted(rows, key=lambda item: item["eval_kl"])


def pick_candidate_layers(ranking: list[dict], base_layers: set[int], count: int) -> list[dict]:
    return [item for item in ranking if item["layer"] not in base_layers][:count]


def kv_specs_for_layers(layers: list[int], ranking_by_layer: dict[int, dict], default_rotation: str) -> list[dict]:
    specs = []
    for layer in layers:
        row = ranking_by_layer.get(layer)
        specs.append({"layer": layer, "rotation": row["rotation"] if row else default_rotation})
    return specs


def patch_kv_specs(model, specs: list[dict], bits: int, alpha: float, seed: int) -> list:
    saved = []
    rotations = sorted({item["rotation"] for item in specs})
    for rotation in rotations:
        layers = [item["layer"] for item in specs if item["rotation"] == rotation]
        saved += patch_attention_layers(model, layers, bits, rotation, alpha, seed)
    return saved


def metrics_delta(metrics: dict, base_metrics: dict) -> dict:
    return {
        "kl": metrics["kl_from_baseline"] - base_metrics["kl_from_baseline"],
        "topk_overlap": metrics["topk_overlap"] - base_metrics["topk_overlap"],
        "loss_delta_from_baseline": metrics["loss_delta_from_baseline"] - base_metrics["loss_delta_from_baseline"],
    }


def memory_estimate(config, kv_specs: list[dict], mlp_params: list[dict], args) -> dict:
    head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    intermediate_size = int(config.intermediate_size)

    kv_full_bits = 2 * args.memory_context_length * num_kv_heads * head_dim * args.full_bits
    kv_quant_bits = 2 * args.memory_context_length * num_kv_heads * (
        head_dim * args.kv_bits + args.scale_bits
    )
    kv_saved_per_layer = max(0.0, (kv_full_bits - kv_quant_bits) / 8.0)

    mlp_full_bits = args.mlp_lifetime_tokens * intermediate_size * args.full_bits
    mlp_quant_bits = args.mlp_lifetime_tokens * (intermediate_size * args.mlp_bits + args.scale_bits)
    mlp_saved_per_layer = max(0.0, (mlp_full_bits - mlp_quant_bits) / 8.0)

    kv_saved = len(kv_specs) * kv_saved_per_layer
    mlp_saved = len(mlp_params) * mlp_saved_per_layer
    total_saved = kv_saved + mlp_saved
    return {
        "context_length": args.memory_context_length,
        "mlp_lifetime_tokens": args.mlp_lifetime_tokens,
        "full_bits": args.full_bits,
        "scale_bits": args.scale_bits,
        "kv_layers": len(kv_specs),
        "mlp_layers": len(mlp_params),
        "kv_saved_bytes_per_layer": kv_saved_per_layer,
        "mlp_saved_bytes_per_layer": mlp_saved_per_layer,
        "kv_saved_mb": kv_saved / (1024 * 1024),
        "mlp_saved_mb": mlp_saved / (1024 * 1024),
        "total_saved_mb": total_saved / (1024 * 1024),
    }


def attach_memory_and_score(item: dict, config, kv_specs: list[dict], mlp_params: list[dict], args) -> dict:
    memory = memory_estimate(config, kv_specs, mlp_params, args)
    item["memory"] = memory
    item["scores"] = {
        "kl": item["metrics"]["kl_from_baseline"],
        "kl_per_saved_mb": item["metrics"]["kl_from_baseline"] / max(memory["total_saved_mb"], 1e-9),
    }
    return item


def summarize_allocator_candidate(item: dict) -> dict:
    summary = {
        "candidate": item["candidate"],
        "kind": item["kind"],
        "layer": item["layer"],
        "independent_eval_kl": item["independent_eval_kl"],
        "marginal_kl": item["delta_vs_base_stack"]["kl"],
        "stack_kl": item["metrics"]["kl_from_baseline"],
        "stack_topk": item["metrics"]["topk_overlap"],
    }
    if "memory" in item:
        summary["saved_mb"] = item["memory"]["total_saved_mb"]
    if "scores" in item:
        summary["kl_per_saved_mb"] = item["scores"]["kl_per_saved_mb"]
    if "delta_vs_current_stack" in item:
        summary["current_marginal_kl"] = item["delta_vs_current_stack"]["kl"]
    return summary


def run_stack(
    model,
    eval_batch,
    baseline_logits,
    baseline_metrics,
    topk: int,
    kv_specs: list[dict],
    kv_bits: int,
    kv_alpha: float,
    mlp_params: list[dict],
    mlp_bits: int,
    mlp_block_size: int,
    seed: int,
) -> dict:
    saved_kv = patch_kv_specs(model, kv_specs, kv_bits, kv_alpha, seed) if kv_specs else []
    saved_mlp = patch_mlp_layers(model, mlp_params, mlp_bits, mlp_block_size, seed) if mlp_params else []
    metrics = summarize_candidate(model, eval_batch, baseline_logits, baseline_metrics, topk)
    restore_forwards(saved_mlp)
    restore_forwards(saved_kv)
    return metrics


def evaluate_addition(
    model,
    eval_batch,
    baseline_logits,
    baseline,
    args,
    current_kv_specs: list[dict],
    current_mlp_params: list[dict],
    current_metrics: dict,
    item: dict,
) -> dict:
    if item["kind"] == "kv":
        kv_specs = current_kv_specs + [{"layer": item["layer"], "rotation": item["rotation"]}]
        mlp_params = current_mlp_params
    else:
        kv_specs = current_kv_specs
        mlp_params = current_mlp_params + [item]

    metrics = run_stack(
        model,
        eval_batch,
        baseline_logits,
        baseline,
        args.topk,
        kv_specs,
        args.kv_bits,
        args.kv_alpha,
        mlp_params,
        args.mlp_bits,
        args.block_size,
        args.seed,
    )
    candidate = {
        "candidate": f"add_{item['kind']}_{item['layer']}",
        "kind": item["kind"],
        "layer": item["layer"],
        "independent_eval_kl": item["eval_kl"],
        "metrics": metrics,
        "delta_vs_base_stack": metrics_delta(metrics, args.base_stack_metrics),
        "delta_vs_current_stack": metrics_delta(metrics, current_metrics),
    }
    if item["kind"] == "kv":
        candidate["rotation"] = item["rotation"]
    else:
        candidate["rotation"] = item["rotation"]
        candidate["alpha"] = item["alpha"]
        if "calibration" in item:
            candidate["calibration"] = item["calibration"]
    return attach_memory_and_score(candidate, model.config, kv_specs, mlp_params, args)


def chosen_to_item(chosen: dict) -> dict:
    item = {
        "layer": chosen["layer"],
        "kind": chosen["kind"],
        "rotation": chosen["rotation"],
        "eval_kl": chosen["independent_eval_kl"],
    }
    if chosen["kind"] == "mlp":
        item["alpha"] = chosen["alpha"]
        if "calibration" in chosen:
            item["calibration"] = chosen["calibration"]
    return item


def selected_keys(kv_specs: list[dict], mlp_params: list[dict]) -> set[tuple[str, int]]:
    return {("kv", item["layer"]) for item in kv_specs} | {("mlp", item["layer"]) for item in mlp_params}


def stack_key(kv_specs: list[dict], mlp_params: list[dict]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(selected_keys(kv_specs, mlp_params)))


def stack_scores(metrics: dict, memory: dict) -> dict:
    return {
        "kl": metrics["kl_from_baseline"],
        "kl_per_saved_mb": metrics["kl_from_baseline"] / max(memory["total_saved_mb"], 1e-9),
    }


def summarize_state(state: dict) -> dict:
    return {
        "kv_layers": [item["layer"] for item in state["kv_specs"]],
        "mlp_layers": [item["layer"] for item in state["mlp_params"]],
        "metrics": state["metrics"],
        "memory": state["memory"],
        "scores": state["scores"],
        "path": [summarize_allocator_candidate(item) for item in state["path"]],
    }


def run_greedy_allocator(
    model,
    eval_batch,
    baseline_logits,
    baseline,
    args,
    base_kv_specs: list[dict],
    base_mlp_params: list[dict],
    base_stack: dict,
    kv_candidates: list[dict],
    mlp_candidates: list[dict],
) -> dict:
    current_kv_specs = list(base_kv_specs)
    current_mlp_params = list(base_mlp_params)
    current_metrics = base_stack
    remaining = [dict(item, kind="kv") for item in kv_candidates] + [dict(item, kind="mlp") for item in mlp_candidates]
    steps = []

    for step_index in range(args.greedy_steps):
        if not remaining:
            break

        evaluated = [
            evaluate_addition(
                model,
                eval_batch,
                baseline_logits,
                baseline,
                args,
                current_kv_specs,
                current_mlp_params,
                current_metrics,
                item,
            )
            for item in remaining
        ]
        evaluated = sorted(evaluated, key=lambda item: item["metrics"]["kl_from_baseline"])
        chosen = evaluated[0]
        steps.append(
            {
                "step": step_index + 1,
                "chosen": chosen,
                "ranked_candidates": evaluated,
            }
        )
        print(
            f"greedy step={step_index + 1} chose={chosen['candidate']} "
            f"stack_kl={chosen['metrics']['kl_from_baseline']:.6f} "
            f"delta={chosen['delta_vs_current_stack']['kl']:.6f}"
        )

        if chosen["kind"] == "kv":
            current_kv_specs.append({"layer": chosen["layer"], "rotation": chosen["rotation"]})
        else:
            current_mlp_params.append(
                {
                    "layer": chosen["layer"],
                    "rotation": chosen["rotation"],
                    "alpha": chosen["alpha"],
                    "calibration": next(
                        item.get("calibration")
                        for item in remaining
                        if item["kind"] == "mlp" and item["layer"] == chosen["layer"]
                    ),
                }
            )
        current_metrics = chosen["metrics"]
        remaining = [item for item in remaining if not (item["kind"] == chosen["kind"] and item["layer"] == chosen["layer"])]

    return {
        "final_kv_specs": current_kv_specs,
        "final_mlp_params": current_mlp_params,
        "final_metrics": current_metrics,
        "steps": steps,
    }


def run_beam_allocator(
    model,
    eval_batch,
    baseline_logits,
    baseline,
    args,
    base_kv_specs: list[dict],
    base_mlp_params: list[dict],
    base_stack: dict,
    kv_candidates: list[dict],
    mlp_candidates: list[dict],
) -> dict:
    pool = [dict(item, kind="kv") for item in kv_candidates] + [dict(item, kind="mlp") for item in mlp_candidates]
    base_memory = memory_estimate(model.config, base_kv_specs, base_mlp_params, args)
    base_state = {
        "kv_specs": list(base_kv_specs),
        "mlp_params": list(base_mlp_params),
        "metrics": base_stack,
        "memory": base_memory,
        "scores": stack_scores(base_stack, base_memory),
        "path": [],
    }
    beam = [base_state]
    frontiers = []

    for step_index in range(args.beam_steps):
        deduped: dict[tuple[tuple[str, int], ...], dict] = {}
        for state in beam:
            chosen = selected_keys(state["kv_specs"], state["mlp_params"])
            remaining = [item for item in pool if (item["kind"], item["layer"]) not in chosen]
            for item in remaining:
                candidate = evaluate_addition(
                    model,
                    eval_batch,
                    baseline_logits,
                    baseline,
                    args,
                    state["kv_specs"],
                    state["mlp_params"],
                    state["metrics"],
                    item,
                )
                if item["kind"] == "kv":
                    kv_specs = state["kv_specs"] + [{"layer": item["layer"], "rotation": item["rotation"]}]
                    mlp_params = state["mlp_params"]
                else:
                    kv_specs = state["kv_specs"]
                    mlp_params = state["mlp_params"] + [item]
                new_state = {
                    "kv_specs": kv_specs,
                    "mlp_params": mlp_params,
                    "metrics": candidate["metrics"],
                    "memory": candidate["memory"],
                    "scores": candidate["scores"],
                    "path": state["path"] + [candidate],
                }
                key = stack_key(kv_specs, mlp_params)
                old = deduped.get(key)
                if old is None or state_sort_key(new_state, args.beam_objective) < state_sort_key(
                    old, args.beam_objective
                ):
                    deduped[key] = new_state

        ranked = sorted(deduped.values(), key=lambda item: state_sort_key(item, args.beam_objective))
        beam = ranked[: args.beam_width]
        frontiers.append(
            {
                "added_buses": step_index + 1,
                "beam": [summarize_state(state) for state in beam],
                "best": summarize_state(beam[0]),
            }
        )
        best = beam[0]
        last = best["path"][-1]
        print(
            f"beam step={step_index + 1} objective={args.beam_objective} "
            f"best_add={last['candidate']} kl={best['metrics']['kl_from_baseline']:.6f} "
            f"saved_mb={best['memory']['total_saved_mb']:.3f} "
            f"score={best['scores'][args.beam_objective]:.6f}"
        )

    return {
        "objective": args.beam_objective,
        "beam_width": args.beam_width,
        "beam_steps": args.beam_steps,
        "final_beam": [summarize_state(state) for state in beam],
        "frontiers": frontiers,
    }


def state_sort_key(state: dict, objective: str) -> tuple[float, float, float]:
    return (
        state["scores"][objective],
        state["metrics"]["kl_from_baseline"],
        -state["memory"]["total_saved_mb"],
    )


def run_probe(args) -> dict:
    kv_payload = load_json(args.kv_impact_result)
    mlp_payload = load_json(args.mlp_impact_result)
    kv_ranking = ranked_kv_layers(kv_payload)
    mlp_ranking = ranked_mlp_layers(mlp_payload)
    kv_by_layer = {item["layer"]: item for item in kv_ranking}
    mlp_by_layer = {item["layer"]: item for item in mlp_ranking}

    base_kv_layers = parse_int_list(args.base_kv_layers)
    base_mlp_layers = parse_int_list(args.base_mlp_layers)
    kv_candidates = pick_candidate_layers(kv_ranking, set(base_kv_layers), args.candidate_count)
    mlp_candidates = pick_candidate_layers(mlp_ranking, set(base_mlp_layers), args.candidate_count)

    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.float32)
    model.eval()

    eval_prompts = load_prompts(args.eval_prompts, EVAL_PROMPTS)
    eval_batch = tokenize(tokenizer, eval_prompts, args.max_length)
    baseline = evaluate_model(model, eval_batch, topk=args.topk)
    baseline_logits = baseline.pop("logits")

    base_kv_specs = kv_specs_for_layers(base_kv_layers, kv_by_layer, args.default_kv_rotation)
    base_mlp_params = [mlp_by_layer[layer] for layer in base_mlp_layers]
    base_stack = run_stack(
        model,
        eval_batch,
        baseline_logits,
        baseline,
        args.topk,
        base_kv_specs,
        args.kv_bits,
        args.kv_alpha,
        base_mlp_params,
        args.mlp_bits,
        args.block_size,
        args.seed,
    )
    print(
        f"base stack kv={base_kv_layers} mlp={base_mlp_layers} "
        f"kl={base_stack['kl_from_baseline']:.6f} topk={base_stack['topk_overlap']:.6f}"
    )
    args.base_stack_metrics = base_stack
    base_memory = memory_estimate(model.config, base_kv_specs, base_mlp_params, args)
    base_scores = stack_scores(base_stack, base_memory)

    candidates = []
    if not args.skip_one_step:
        for item in kv_candidates:
            metrics = evaluate_addition(
                model,
                eval_batch,
                baseline_logits,
                baseline,
                args,
                base_kv_specs,
                base_mlp_params,
                base_stack,
                dict(item, kind="kv"),
            )
            candidates.append(metrics)
            print(
                f"add kv layer={item['layer']} independent_kl={item['eval_kl']:.6f} "
                f"stack_kl={metrics['metrics']['kl_from_baseline']:.6f} "
                f"delta={metrics['delta_vs_base_stack']['kl']:.6f}"
            )

        for item in mlp_candidates:
            metrics = evaluate_addition(
                model,
                eval_batch,
                baseline_logits,
                baseline,
                args,
                base_kv_specs,
                base_mlp_params,
                base_stack,
                dict(item, kind="mlp"),
            )
            candidates.append(metrics)
            print(
                f"add mlp layer={item['layer']} independent_kl={item['eval_kl']:.6f} "
                f"stack_kl={metrics['metrics']['kl_from_baseline']:.6f} "
                f"delta={metrics['delta_vs_base_stack']['kl']:.6f}"
            )

    greedy = None
    if args.greedy_steps:
        greedy = run_greedy_allocator(
            model,
            eval_batch,
            baseline_logits,
            baseline,
            args,
            base_kv_specs,
            base_mlp_params,
            base_stack,
            kv_candidates,
            mlp_candidates,
        )

    beam = None
    if args.beam_steps:
        beam = run_beam_allocator(
            model,
            eval_batch,
            baseline_logits,
            baseline,
            args,
            base_kv_specs,
            base_mlp_params,
            base_stack,
            kv_candidates,
            mlp_candidates,
        )

    by_marginal_kl = sorted(candidates, key=lambda item: item["delta_vs_base_stack"]["kl"])
    by_independent_kl = sorted(candidates, key=lambda item: item["independent_eval_kl"])
    return {
        "experiment": "joint_allocator_probe",
        "repo": args.repo,
        "kv_impact_result": str(args.kv_impact_result),
        "mlp_impact_result": str(args.mlp_impact_result),
        "kv_bits": args.kv_bits,
        "kv_alpha": args.kv_alpha,
        "mlp_bits": args.mlp_bits,
        "block_size": args.block_size,
        "eval_prompt_count": len(eval_prompts),
        "baseline_eval": baseline,
        "base_kv_specs": base_kv_specs,
        "base_mlp_params": base_mlp_params,
        "base_stack": base_stack,
        "base_memory": base_memory,
        "base_scores": base_scores,
        "kv_candidates": kv_candidates,
        "mlp_candidates": mlp_candidates,
        "candidates": candidates,
        "greedy": greedy,
        "beam": beam,
        "summary": {
            "best_by_marginal_kl": [summarize_allocator_candidate(item) for item in by_marginal_kl],
            "best_by_independent_kl": [summarize_allocator_candidate(item) for item in by_independent_kl],
            "greedy_path": [
                summarize_allocator_candidate(step["chosen"]) for step in greedy["steps"]
            ]
            if greedy
            else [],
            "beam_frontier": [
                {
                    "added_buses": item["added_buses"],
                    "best": {
                        "kv_layers": item["best"]["kv_layers"],
                        "mlp_layers": item["best"]["mlp_layers"],
                        "kl": item["best"]["metrics"]["kl_from_baseline"],
                        "topk": item["best"]["metrics"]["topk_overlap"],
                        "loss_delta": item["best"]["metrics"]["loss_delta_from_baseline"],
                        "saved_mb": item["best"]["memory"]["total_saved_mb"],
                        "score": item["best"]["scores"][args.beam_objective],
                    },
                }
                for item in beam["frontiers"]
            ]
            if beam
            else [],
        },
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_joint_allocator_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Interaction-aware marginal allocator probe for fused KV + MLP compression")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--kv-impact-result", type=Path, required=True)
    parser.add_argument("--mlp-impact-result", type=Path, required=True)
    parser.add_argument("--base-kv-layers", default="28,24,26,14")
    parser.add_argument("--base-mlp-layers", default="16,17,15,12")
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--skip-one-step", action="store_true")
    parser.add_argument("--greedy-steps", type=int, default=0)
    parser.add_argument("--beam-steps", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--beam-objective", choices=["kl", "kl_per_saved_mb"], default="kl")
    parser.add_argument("--default-kv-rotation", default="hadamard")
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
    parser.add_argument("--eval-prompts")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    result = run_probe(args)
    path = save_result(result, args.out_dir)
    print(json.dumps({"base_stack": result["base_stack"], "summary": result["summary"]}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
