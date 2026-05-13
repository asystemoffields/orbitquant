import argparse
import json
import time
import types
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smol135_full_probe import DEFAULT_REPO, compare_logits, evaluate_model, get_layer, next_token_metrics, tokenize
from smol135_mlp_activation_rotation_probe import capture_mlp_intermediate, rel_mse, rotation_matrix
from smol135_sweep import CALIBRATION_PROMPTS, EVAL_PROMPTS, parse_float_list, parse_int_list, parse_str_list


def quantize_with_rotation(x: torch.Tensor, bits: int, rotation: torch.Tensor | None, alpha: float) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    if rotation is None:
        rotated = x
    else:
        rotation = rotation.to(device=x.device, dtype=x.dtype)
        rotated = x @ rotation
    scale = rotated.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) * alpha / qmax
    quantized = torch.round(rotated / scale).clamp(-qmax, qmax) * scale
    if rotation is None:
        return quantized
    return quantized @ rotation.T


def evaluate_boundary_candidate(z, y, down_weight, bits, rotation, alpha) -> dict:
    z_quant = quantize_with_rotation(z, bits, rotation, alpha)
    y_quant = z_quant @ down_weight.T
    return {
        "activation_rel_mse": rel_mse(z, z_quant),
        "down_output_rel_mse": rel_mse(y, y_quant),
    }


def calibrate_layer_params(args, model, cal_batch, layers: list[int]) -> list[dict]:
    rotations = parse_str_list(args.rotations)
    alphas = parse_float_list(args.alphas)
    params = []

    for layer_idx in layers:
        layer = get_layer(model, layer_idx)
        down_weight = layer.mlp.down_proj.weight.detach().float().cpu()
        z, y = capture_mlp_intermediate(model, cal_batch, layer_idx)

        best = None
        for rotation_name in rotations:
            rot = None
            if rotation_name != "identity":
                rot = rotation_matrix(
                    z.shape[-1],
                    rotation_name,
                    args.block_size,
                    args.seed + layer_idx * 1009 + len(rotation_name),
                )
            for alpha in alphas:
                metrics = evaluate_boundary_candidate(z, y, down_weight, args.bits, rot, alpha)
                candidate = {
                    "layer": layer_idx,
                    "rotation": rotation_name,
                    "alpha": alpha,
                    "calibration": metrics,
                }
                if best is None or metrics["down_output_rel_mse"] < best["calibration"]["down_output_rel_mse"]:
                    best = candidate

        params.append(best)
        print(
            f"cal layer={layer_idx} best={best['rotation']} alpha={best['alpha']:.3f} "
            f"cal_down={best['calibration']['down_output_rel_mse']:.6f}"
        )

    return params


def make_mlp_forward(mlp, bits: int, rotation: torch.Tensor | None, alpha: float):
    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        z = gate * up
        z_quant = quantize_with_rotation(z, bits, rotation, alpha)
        return self.down_proj(z_quant)

    return types.MethodType(forward, mlp)


def patch_mlp_layers(model, layer_params: list[dict], bits: int, block_size: int, seed: int) -> list:
    saved = []
    for item in layer_params:
        layer_idx = item["layer"]
        layer = get_layer(model, layer_idx)
        original_forward = layer.mlp.forward
        rotation = None
        if item["rotation"] != "identity":
            rotation = rotation_matrix(
                layer.mlp.down_proj.weight.shape[1],
                item["rotation"],
                block_size,
                seed + layer_idx * 1009 + len(item["rotation"]),
            )
        layer.mlp.forward = make_mlp_forward(layer.mlp, bits, rotation, item["alpha"])
        saved.append((layer.mlp, original_forward))
    return saved


def restore_forwards(saved: list) -> None:
    for module, original_forward in saved:
        module.forward = original_forward


def summarize_candidate(model, batch, baseline_logits, baseline_metrics, topk) -> dict:
    metrics = evaluate_model(model, batch, baseline_logits, topk)
    metrics.pop("logits")
    metrics["loss_delta_from_baseline"] = metrics["loss"] - baseline_metrics["loss"]
    return metrics


def run_probe(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.float32)
    model.eval()

    layers = parse_int_list(args.layers)
    cal_batch = tokenize(tokenizer, CALIBRATION_PROMPTS, args.max_length)
    eval_batch = tokenize(tokenizer, EVAL_PROMPTS, args.max_length)

    baseline = evaluate_model(model, eval_batch, topk=args.topk)
    baseline_logits = baseline.pop("logits")

    calibrated_params = calibrate_layer_params(args, model, cal_batch, layers)
    identity_params = [{"layer": layer, "rotation": "identity", "alpha": 1.0} for layer in layers]
    clipped_identity_params = [{"layer": layer, "rotation": "identity", "alpha": args.identity_alpha} for layer in layers]

    candidates = []
    for name, params in [
        ("identity_absmax_ternary", identity_params),
        (f"identity_alpha_{args.identity_alpha:g}", clipped_identity_params),
        ("calibrated_rotation_ternary", calibrated_params),
    ]:
        saved = patch_mlp_layers(model, params, args.bits, args.block_size, args.seed)
        metrics = summarize_candidate(model, eval_batch, baseline_logits, baseline, args.topk)
        restore_forwards(saved)
        candidates.append({"candidate": name, "metrics": metrics})
        print(
            f"eval {name} loss={metrics['loss']:.6f} "
            f"kl={metrics['kl_from_baseline']:.6f} topk={metrics['topk_overlap']:.6f}"
        )

    return {
        "experiment": "mlp_stack_probe",
        "repo": args.repo,
        "layers": layers,
        "bits": args.bits,
        "block_size": args.block_size,
        "calibration_prompt_count": len(CALIBRATION_PROMPTS),
        "eval_prompt_count": len(EVAL_PROMPTS),
        "baseline_eval": baseline,
        "calibrated_layer_params": calibrated_params,
        "candidates": candidates,
    }


def save_result(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"smol135_mlp_stack_{payload['bits']}bit_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Smol135 MLP activation quantization stack probe")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--layers", default="0,1,2,4,8,12,16,20,24,29")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--rotations", default="identity,block_hadamard,block_hadamard_sign_perm")
    parser.add_argument("--alphas", default="1.0,0.75,0.5,0.375,0.25")
    parser.add_argument("--identity-alpha", type=float, default=0.375)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    result = run_probe(args)
    path = save_result(result, args.out_dir)
    summary = {
        "baseline_eval": result["baseline_eval"],
        "candidates": result["candidates"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
