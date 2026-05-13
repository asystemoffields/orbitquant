# Smol135 Fused Activation Policy

Base model: `HuggingFaceTB/SmolLM2-135M`

This artifact records a fused activation-compression policy for SmolLM2-135M. It packages the selected KV-cache and MLP intermediate-bus transforms, the source result path, and the held-out evaluation metrics used to choose the policy.

## Selected Result

| Metric | Value |
|---|---:|
| Total compressed buses | 8 |
| KL from baseline | 0.057946 |
| Top-k overlap | 0.856197 |
| Loss delta | -0.035715 |
| Estimated saved MB | 5.43701171875 |

## KV Policy

| layer | bits | rotation | alpha |
| --- | --- | --- | --- |
| 28 | 3 | hadamard | 0.75 |
| 24 | 3 | hadamard | 0.75 |
| 26 | 3 | hadamard | 0.75 |
| 14 | 3 | hadamard | 0.75 |

## MLP Policy

| layer | bits | primitive | rotation | alpha | block_size |
| --- | --- | --- | --- | --- | --- |
| 16 | 2 | plain | block_hadamard | 0.375 | 512 |
| 17 | 2 | plus | preperm_activation_max_hadamard | 0.375 | 512 |
| 15 | 2 | plain | block_hadamard | 0.375 | 512 |
| 12 | 2 | plain | block_hadamard | 0.375 | 512 |

## Evaluation

Calibration prompts: `prompts/broad_calibration.txt`

Held-out eval prompts: `prompts/broad_eval.txt`

Max sequence length: `64`

Stopped reason: `None`

## Files

- `compression_config.json`: runtime policy and metrics.
- `manifest.json`: compact artifact summary.
- `README.md`: model-card draft for publication.
