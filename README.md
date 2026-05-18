# OrbitQuant

OrbitQuant explores allocation-aware quantization over equivalent activation bases. The current prototype compresses selected SmolLM2-135M runtime buses by stacking:

- Hadamard-rotated 3-bit KV-cache activation quantization.
- Rotated/clipped 2-bit MLP intermediate activation quantization.
- A fused allocator that chooses layers and MLP primitives under held-out logit drift.

Current best practical artifacts:

```text
hf_artifacts/smol135-fused-policy-quality16-14bus
hf_artifacts/gemma4-pmra-orbitquant-safe3
```

Published Gemma4 policy artifact:

```text
https://huggingface.co/Asystemoffields/gemma4-pmra-orbitquant-safe3
```

## Gemma4 PMRA Stack Check

The Gemma4 stack check applies OrbitQuant runtime activation compression on top of the PMRA Gemma4 E2B-it knapsack artifact. It validates that the runtime OrbitQuant hooks and the PMRA-patched weight state compose in one model-forward evaluator.

Result note:

```text
notes/gemma4_pmra_orbit_stack_result.md
notes/gemma4_pmra_orbit_damage_attribution.md
notes/orbitquant_production_optimization_map.md
```

Fetched Modal summary:

```text
results/modal_gemma4_pmra_orbit_stack_stack64_latest.json
results/modal_gemma4_pmra_orbit_stack_split64_latest.json
results/modal_gemma4_pmra_orbit_layer_mlp_split64_latest.json
results/modal_gemma4_pmra_orbit_stack_trim64_safe3_latest.json
results/modal_gemma4_pmra_orbit_stack_trim128_safe3_latest.json
```

Current Gemma4 PMRA + OrbitQuant operating point on Wikitext test, 128 prompts / 24,058 tokens:

| Variant | NLL | Payload bpw |
|---|---:|---:|
| q3_k_s | 18.046307 | 5.326613 |
| PMRA | 12.818462 | 5.326613 |
| PMRA + KV-only OrbitQuant | 12.874908 | 5.326613 |
| PMRA + OrbitQuant safe3 | 12.834083 | 5.326613 |

The `safe3` stack keeps PMRA's static payload and adds a runtime overlay:

- KV layers: `33,28,30,16,18,11,15`, 3-bit Hadamard.
- MLP buses: `20:plus:preperm_activation_max_hadamard`, `19:plus:preperm_activation_max_hadamard`, `6:plus:preperm_boundary_rms_hadamard`, 2-bit.
- Estimated runtime memory saved: `48.78 MiB` at context length 8192 with 64 live MLP tokens.
- NLL cost over PMRA: `0.015620`.

The damage attribution run found that the original inherited MLP buses caused most of the stack loss. Dropping layers 14/15 recovered `0.375819` NLL; trimming to the safe three MLP buses recovered `0.597284` NLL versus the original split stack. The 128-prompt validation tightened the safe3 estimate from `+0.075261` to `+0.015620` NLL over PMRA.

Export the Gemma4 policy artifact:

```powershell
python scripts/export_gemma4_orbit_policy_artifact.py --result results\modal_gemma4_pmra_orbit_stack_trim128_safe3_latest.json --name gemma4-pmra-orbitquant-safe3
```

Dry-run the packaged policy:

```powershell
python scripts/apply_gemma4_orbit_policy.py --config hf_artifacts\gemma4-pmra-orbitquant-safe3\compression_config.json --dry-run
```

Quality16 14-bus result on the broad held-out prompt split:

| Metric | Value |
|---|---:|
| KV layers compressed | 7 |
| MLP intermediate buses compressed | 7 |
| KL from baseline | 0.110638 |
| Top-k overlap | 0.814085 |
| Loss delta | -0.026857 |
| Estimated saved memory | 9.515 MiB |

The original scratchpad began with SmolLM3 tensor-orbit probes. Those early probes remain below because they show how the activation-basis idea developed.

## Early SmolLM3 Probe

Small experiments for testing representation-aware compression ideas on `HuggingFaceTB/SmolLM3-3B`.

The first probe works directly with safetensor shards, loading only selected tensors into memory. That keeps the workflow useful on CPU machines while still touching the real SmolLM3 weights.

## Current Probe

`scripts/smollm3_orbit_probe.py` tests exact basis changes before quantization:

- **Value/O attention orbit:** rotate one GQA value head and the O-projection input blocks that consume it.
- **MLP up/down scaling orbit:** rescale SwiGLU `up_proj` hidden channels and inversely rescale `down_proj` columns.

Run the lighter attention probe:

```powershell
python scripts/smollm3_orbit_probe.py --experiment value_o --layer 0 --bits 4 --trials 6
```

Run both probes:

```powershell
python scripts/smollm3_orbit_probe.py --experiment both --layer 0 --bits 4 --trials 4
```

Results are written under `results/` as JSON.

## Full-Model Smol135 Probe

`scripts/smol135_full_probe.py` defaults to `HuggingFaceTB/SmolLM2-135M`, which is small enough to load fully on CPU. It measures actual next-token loss, logit KL, and top-k overlap after applying a compressed edit.

Run the combined value/O and typical-subspace probe:

```powershell
python scripts/smol135_full_probe.py --experiment both --layer 0 --bits 3
```

Run only the typical-subspace probe with a 2.2-bit effective input-direction budget:

```powershell
python scripts/smol135_full_probe.py --experiment typical_vproj --layer 0 --bits 3 --typical-hi-bits 4 --typical-lo-bits 2 --typical-top-frac 0.10
```

Try the original SmolLM 135M checkpoint:

```powershell
python scripts/smol135_full_probe.py --repo HuggingFaceTB/SmolLM-135M --experiment both --layer 0 --bits 3
```

Run held-out sweeps across layers:

```powershell
python scripts/smol135_sweep.py --experiment value_o --layers 0,1,2,4,8,12,16,20,24,29 --bits 2 --rotations identity,hadamard,hadamard_sign_perm,random_orthogonal
```

Run the TurboQuant-style KV activation rotation probe:

```powershell
$layers = (0..29) -join ','
python scripts/smol135_kv_rotation_probe.py --layers $layers --bits 3 --rotations identity,hadamard,random_orthogonal
```

Run the MLP intermediate activation rotation probe:

```powershell
$layers = (0..29) -join ','
python scripts/smol135_mlp_activation_rotation_probe.py --layers $layers --bits 3 --rotations identity,block_hadamard,block_hadamard_sign_perm --block-size 512
```

Run the calibrated 2-bit MLP ternary scale search:

```powershell
$layers = (0..29) -join ','
python scripts/smol135_mlp_ternary_search.py --layers $layers --bits 2 --rotations identity,block_hadamard,block_hadamard_sign_perm --alphas 1.0,0.75,0.5,0.375,0.25 --block-size 512
```

Run an end-to-end selected-layer 2-bit MLP stack:

```powershell
python scripts/smol135_mlp_stack_probe.py --layers 16,17,12,6,14,21,5,19 --bits 2 --rotations identity,block_hadamard,block_hadamard_sign_perm --alphas 1.0,0.75,0.5,0.375,0.25 --identity-alpha 0.375 --block-size 512
```

Run the per-layer impact sweep and automatic layer allocator:

```powershell
$layers = (0..29) -join ','
python scripts/smol135_mlp_impact_sweep.py --layers $layers --bits 2 --rotations identity,block_hadamard,block_hadamard_sign_perm --alphas 1.0,0.75,0.5,0.375,0.25 --thresholds 0.005,0.01,0.02,0.05,0.10,0.20 --top-counts 4,8,12,16,20 --block-size 512
```

Replicate the allocator on the original SmolLM 135M checkpoint:

```powershell
$layers = (0..29) -join ','
python scripts/smol135_mlp_impact_sweep.py --repo HuggingFaceTB/SmolLM-135M --layers $layers --bits 2 --rotations identity,block_hadamard,block_hadamard_sign_perm --alphas 1.0,0.75,0.5,0.375,0.25 --thresholds 0.01,0.02,0.05,0.10 --top-counts 4,8,12 --block-size 512
```

Run the fused KV + MLP stack probe:

```powershell
python scripts/smol135_fused_stack_probe.py --kv-layers 28,8,24,15 --kv-bits 3 --kv-rotation hadamard --kv-alpha 0.75 --mlp-layers 16,17,12,6 --mlp-bits 2 --block-size 512
```

Run the KV per-layer allocator:

```powershell
python scripts/smol135_kv_impact_sweep.py --bits 3 --rotations identity,hadamard --top-counts 4,8,12,16
```

Run broad-prompt fused validation:

```powershell
python scripts/smol135_fused_stack_probe.py --calibration-prompts prompts/broad_calibration.txt --eval-prompts prompts/broad_eval.txt --kv-layers 28,24,26,14 --kv-bits 3 --kv-rotation hadamard --kv-alpha 0.75 --mlp-layers 16,17,15,12 --mlp-bits 2 --block-size 512 --max-length 64
```

Run the interaction-aware joint allocator probe:

```powershell
python scripts/smol135_joint_allocator_probe.py --kv-impact-result results\smol135_kv_impact_3bit_20260513_084903.json --mlp-impact-result results\smol135_mlp_impact_2bit_20260513_084037.json --base-kv-layers 28,24,26,14 --base-mlp-layers 16,17,15,12 --candidate-count 8 --greedy-steps 4 --kv-bits 3 --kv-alpha 0.75 --mlp-bits 2 --block-size 512 --eval-prompts prompts\broad_eval.txt --max-length 64
```

Run the prepared Hadamard-plus MLP bus search:

```powershell
$layers = (0..29) -join ','
python scripts/smol135_mlp_hadamard_plus_probe.py --layers $layers --bits 2 --alphas 1.0,0.75,0.5,0.375,0.25 --block-size 512 --calibration-prompts prompts\broad_calibration.txt --eval-prompts prompts\broad_eval.txt --max-length 64
```

Run Hadamard-plus end-to-end MLP impact from a saved local result:

```powershell
python scripts/smol135_mlp_hadamard_plus_impact_sweep.py --local-result results\smol135_mlp_hadamard_plus_2bit_20260513_131208.json --bits 2 --top-counts 4,8,12 --block-size 512 --calibration-prompts prompts\broad_calibration.txt --eval-prompts prompts\broad_eval.txt --max-length 64
```

Run a fused KV + Hadamard-plus MLP check:

```powershell
python scripts/smol135_fused_hadamard_plus_probe.py --local-result results\smol135_mlp_hadamard_plus_2bit_20260513_131208.json --kv-layers 28,24,26,14 --kv-bits 3 --kv-rotation hadamard --kv-alpha 0.75 --mlp-layers 17,16,15,13 --mlp-bits 2 --block-size 512 --calibration-prompts prompts\broad_calibration.txt --eval-prompts prompts\broad_eval.txt --max-length 64
```

Run the fused primitive-choice allocator:

```powershell
python scripts/smol135_fused_choice_allocator.py --kv-impact-result results\smol135_kv_impact_3bit_20260513_084903.json --local-result results\smol135_mlp_hadamard_plus_2bit_20260513_131208.json --mlp-choice-impact-result results\smol135_mlp_hadamard_plus_impact_2bit_20260513_132208.json --base-kv-layers 28,24,26,14 --base-mlp-choices 16:plain,17:plain,15:plain,12:plain --candidate-count 8 --beam-steps 8 --beam-width 3 --kv-bits 3 --kv-alpha 0.75 --mlp-bits 2 --block-size 512 --calibration-prompts prompts\broad_calibration.txt --eval-prompts prompts\broad_eval.txt --max-length 64
```

Run targeted primitive swaps on a fixed fused frontier:

```powershell
python scripts/smol135_fused_primitive_swap_probe.py --local-result results\smol135_mlp_hadamard_plus_2bit_20260513_131208.json --kv-layers 28,24,26,14,13,8,17 --kv-bits 3 --kv-alpha 0.75 --kv-rotation hadamard --mlp-layers 16,17,15,12,13,19,6,14,5 --mlp-bits 2 --block-size 512 --calibration-prompts prompts\broad_calibration.txt --eval-prompts prompts\broad_eval.txt --max-length 64 --pair-search
```

Run the fused policy search on Modal:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
modal run --detach --timestamps --write-result results\modal_runs\fused-quality16.result.txt modal_fused_policy_search.py --profile quality16
```

For a true background Modal function call that survives local client disconnects:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
modal run --detach --timestamps --write-result results\modal_runs\fused-quality16-spawn.result.json modal_fused_policy_search.py --profile quality16 --background
```

Export a completed policy-search result as a HF-style artifact:

```powershell
python scripts/export_hf_policy_artifact.py --result results\modal_smol135_fused_policy_search_20260513_192003.json --name smol135-fused-policy-smoke
```

Export specific frontier budgets from the completed `quality16` run:

```powershell
python scripts/export_hf_policy_artifact.py --result results\modal_smol135_fused_policy_search_20260513_211243.json --name smol135-fused-policy-quality16-8bus --target-buses 8
python scripts/export_hf_policy_artifact.py --result results\modal_smol135_fused_policy_search_20260513_211243.json --name smol135-fused-policy-quality16-14bus --target-buses 14
python scripts/export_hf_policy_artifact.py --result results\modal_smol135_fused_policy_search_20260513_211243.json --name smol135-fused-policy-quality16-16bus --target-buses 16
```

Dry-run a packaged policy before loading the model:

```powershell
python scripts/apply_fused_policy.py --config hf_artifacts\smol135-fused-policy-quality16-16bus\compression_config.json --dry-run
```

Fetch the latest persisted Modal result after a background run completes:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
modal run --write-result results\modal_runs\fused-quality16-latest.json modal_fused_policy_search.py::read_latest_persisted --profile quality16
```
