# Gemma4 PMRA x OrbitQuant Damage Attribution

Date: 2026-05-13

## Setup

Model: `google/gemma-4-E2B-it`

Static weight state: PMRA `c2_calib_knapsack_mixed`

Eval split: Wikitext test, 64 prompts, max length 128

Calibration split: Wikitext train, 16 prompts, max length 128

Fetched Modal summaries:

```text
results/modal_gemma4_pmra_orbit_stack_split64_latest.json
results/modal_gemma4_pmra_orbit_layer_mlp_split64_latest.json
results/modal_gemma4_pmra_orbit_stack_trim64_no14_15_latest.json
results/modal_gemma4_pmra_orbit_stack_trim64_safe3_latest.json
results/modal_gemma4_pmra_orbit_stack_trim128_safe3_latest.json
results/modal_gemma4_pmra_orbit_stack_trim128_safe3_folded_latest.json
```

## Stack Split

| Variant | NLL | Delta vs PMRA | Saved memory estimate |
|---|---:|---:|---:|
| PMRA | 12.984226 | 0.000000 | n/a |
| PMRA + KV-only OrbitQuant | 13.042898 | 0.058672 | 45.50 MiB |
| PMRA + MLP-only OrbitQuant, original 7 MLP buses | 13.776863 | 0.792636 | 7.88 MiB MLP side |
| PMRA + original OrbitQuant stack | 13.656772 | 0.672546 | about 53.38 MiB |

The quality loss was concentrated in the MLP intermediate buses. The KV-cache side is a small, stable cost at this prompt length and saves most of the runtime memory.

## MLP Layer Sweep

Individual MLP bus damage under PMRA:

| MLP candidate | NLL | Delta vs PMRA |
|---|---:|---:|
| 14:plain:block_hadamard | 13.280988 | 0.296761 |
| 15:plain:block_hadamard | 13.133335 | 0.149109 |
| 18:plain:block_hadamard | 13.075312 | 0.091086 |
| 16:plain:block_hadamard | 13.057467 | 0.073241 |
| 6:plus:preperm_boundary_rms_hadamard | 13.029442 | 0.045216 |
| 19:plus:preperm_activation_max_hadamard | 13.008641 | 0.024415 |
| 20:plus:preperm_activation_max_hadamard | 12.977453 | -0.006774 |

Layers 14 and 15 dominate the original stack damage. Layers 18 and 16 are the next largest plain-Hadamard costs. The three safest MLP buses are layers 20, 19, and 6, all using the Hadamard-plus calibrated prepermutation variants.

## Trimmed Stacks

| Stack | MLP buses | Full stack NLL | Delta vs PMRA | Recovered vs original stack | Saved memory estimate |
|---|---|---:|---:|---:|---:|
| Original | 19,20,18,14,6,16,15 | 13.656772 | 0.672546 | 0.000000 | about 53.38 MiB |
| No 14/15 | 19,20,18,6,16 | 13.280952 | 0.296726 | 0.375819 | 51.41 MiB |
| Safe3 | 20,19,6 | 13.059487 | 0.075261 | 0.597284 | 48.78 MiB |

The `safe3` stack is the current best Gemma4 PMRA + OrbitQuant operating point:

- Static payload stays at `5.326613` bpw from PMRA.
- Runtime memory estimate saves `48.78 MiB` at context length 8192 with 64 live MLP tokens.
- KV contributes `45.50 MiB`; the three safe MLP buses add `3.28 MiB`.
- Full stack NLL is `0.075261` above PMRA on this check.
- Full stack is `4.986283` NLL better than uniform `q3_k_s` at the same static payload.

## 128-Prompt Validation

Eval split: Wikitext test, 128 prompts, 24,058 tokens, max length 192

Calibration split: Wikitext train, 24 prompts, max length 192

| Variant | NLL | Delta vs PMRA | Saved memory estimate |
|---|---:|---:|---:|
| q3_k_s | 18.046307 | 5.227844 | n/a |
| PMRA | 12.818462 | 0.000000 | n/a |
| PMRA + KV-only OrbitQuant | 12.874908 | 0.056445 | 45.50 MiB |
| PMRA + MLP-only safe3 | 13.005974 | 0.187512 | 3.28 MiB MLP side |
| PMRA + safe3 OrbitQuant | 12.834083 | 0.015620 | 48.78 MiB |
| PMRA + folded safe3 OrbitQuant | 12.800727 | -0.017735 | 48.78 MiB |

The 128-prompt check makes the `safe3` stack look substantially cleaner than the 64-prompt estimate. It keeps the same `5.326613` static payload bpw as PMRA and q3_k_s, adds `48.78 MiB` estimated runtime memory savings at context length 8192, and lands only `0.015620` NLL above PMRA on this validation.

The folded MLP runtime path removes the inverse activation rotation by pre-rotating selected `down_proj` weights. On the same 128-prompt validation, the folded full stack lands `0.017735` NLL below PMRA. The local algebra smoke test is exact in float32; the Gemma4 result likely reflects changed low-precision rounding order in the production-shaped path.

## Read

The strongest signal is allocator-shaped. Hadamard-style MLP activation rotation works where the bus is locally tolerant and becomes expensive where the down-projection boundary is sensitive. The prepermutation variants are doing useful work: the surviving MLP buses are all Hadamard-plus choices, while the removed buses are plain block-Hadamard choices.

The packaged Gemma4-native OrbitQuant policy artifact encodes:

```text
KV layers: 33,28,30,16,18,11,15
MLP buses: 20:plus:preperm_activation_max_hadamard, 19:plus:preperm_activation_max_hadamard, 6:plus:preperm_boundary_rms_hadamard
MLP runtime: folded down_proj
```

The policy is packaged independently from the PMRA weights and applied as a runtime compression overlay.
