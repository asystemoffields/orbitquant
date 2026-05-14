---
base_model: google/gemma-4-E2B-it
library_name: transformers
tags:
  - orbitquant
  - quantization
  - activation-compression
  - pmra
  - gemma4
pipeline_tag: text-generation
---

# Gemma4 PMRA OrbitQuant Safe3 Policy

Base model: `google/gemma-4-E2B-it`

This artifact records the current Gemma4 OrbitQuant runtime overlay evaluated on top of the PMRA `c2_calib_knapsack_mixed` static weight state.

## Selected Result

| Metric | Value |
|---|---:|
| Total compressed buses | 10 |
| PMRA NLL | 12.818462 |
| Stack NLL | 12.834083 |
| Delta NLL vs PMRA | 0.015620 |
| Delta NLL vs q3_k_s | -5.212224 |
| Estimated saved MiB | 48.78125 |

## KV Policy

| layer | bits | rotation | alpha |
| --- | --- | --- | --- |
| 33 | 3 | hadamard | 0.75 |
| 28 | 3 | hadamard | 0.75 |
| 30 | 3 | hadamard | 0.75 |
| 16 | 3 | hadamard | 0.75 |
| 18 | 3 | hadamard | 0.75 |
| 11 | 3 | hadamard | 0.75 |
| 15 | 3 | hadamard | 0.75 |

## MLP Policy

| layer | bits | primitive | rotation | alpha | block_size |
| --- | --- | --- | --- | --- | --- |
| 20 | 2 | plus | preperm_activation_max_hadamard | 0.375 | 512 |
| 19 | 2 | plus | preperm_activation_max_hadamard | 0.375 | 512 |
| 6 | 2 | plus | preperm_boundary_rms_hadamard | 0.375 | 512 |

## Evaluation

Tokens: `24058`

Prompt count: `128`

Calibration prompt count: `24`

Eval max length: `192`

Calibration max length: `192`

Top-10 overlap vs FP16: `0.13593750000000002`

Last-logit MSE vs FP16: `67.99472899734974`

## Files

- `compression_config.json`: runtime policy and metrics.
- `manifest.json`: compact artifact summary.
- `README.md`: model-card draft for publication.
