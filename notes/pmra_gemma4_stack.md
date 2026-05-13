# PMRA + OrbitQuant Stack Analysis

Date: 2026-05-13

PMRA target: Gemma 4 E2B-it, `c2_calib_knapsack_mixed`

PMRA artifact:

```text
C:\Users\power\Documents\PMRA\tmp\run010_gemma4_knapsack_artifact\gemma4_e2b_it_pmra_calib_knapsack.gguf
```

PMRA result files:

```text
C:\Users\power\Documents\PMRA\results\gemma4_e2b_it\selector_result_knapsack.json
C:\Users\power\Documents\PMRA\results\gemma4_e2b_it\artifact_report_knapsack.json
```

## Relationship

PMRA and OrbitQuant act on different memory surfaces.

PMRA is static tensor-level mixed production quantization:

```text
weight tensor_i := payload_from(selected_GGUF_source_i)
```

The Gemma4 knapsack artifact is one standard GGUF file built from mixed tensor payloads. It needs no custom runtime path in llama.cpp because each tensor already has a production GGUF quantization type.

OrbitQuant is runtime activation-bus compression:

```text
KV activations -> rotate -> low-bit store -> inverse/consume
MLP intermediate activations -> rotate -> low-bit store -> inverse -> down/projection path
```

The stack is therefore conceptually clean:

```text
PMRA reduces resident weight payload.
OrbitQuant reduces runtime KV and selected activation-bus payload.
```

## Gemma4 PMRA Baseline

From `selector_result_knapsack.md`:

| Variant | NLL | Payload bpw | Payload bytes |
|---|---:|---:|---:|
| fp16 | 14.381222 | 16.000000 | 9,294,899,782 |
| Q2_K | 20.376913 | 5.118105 | 2,973,267,084 |
| Q3_K_S target | 17.993582 | 5.326613 | 3,094,396,044 |
| PMRA knapsack | 12.878809 | 5.326613 | 3,094,396,044 |
| same-budget random | 20.488594 | 5.326613 | 3,094,396,044 |

The materialized GGUF payload is `3,094,397,068` bytes and the file size is `3,110,215,968` bytes.

## Gemma4 Shape

Gemma 4 E2B-it text config:

| Field | Value |
|---|---:|
| Text layers | 35 |
| Hidden size | 1536 |
| Attention heads | 8 |
| KV heads | 1 |
| Head dim | 256 |
| Intermediate size | 6144 |
| Sliding window | 512 |
| Max position embeddings | 131072 |

Important wrinkle: the model alternates sliding attention and full attention. KV-cache savings depend on whether the runtime stores all layers at full context length or caps sliding layers at the local window.

## Estimated OrbitQuant Additive Runtime Savings

Using the same accounting as the SmolLM2 policy search: fp16 baseline buffers, 3-bit KV, 2-bit MLP intermediate, 16-bit scales, and `64` live MLP tokens.

For a 14-bus analog with 7 KV layers and 7 MLP buses:

| Context | KV saved | MLP saved | Total saved |
|---:|---:|---:|---:|
| 512 | 2.83 MiB | 4.59 MiB | 7.42 MiB |
| 2,048 | 11.32 MiB | 4.59 MiB | 15.91 MiB |
| 8,192 | 45.28 MiB | 4.59 MiB | 49.87 MiB |
| 32,768 | 181.12 MiB | 4.59 MiB | 185.72 MiB |
| 131,072 | 724.50 MiB | 4.59 MiB | 729.09 MiB |

For a 28-bus analog with 14 KV layers and 14 MLP buses:

| Context | KV saved | MLP saved | Total saved |
|---:|---:|---:|---:|
| 512 | 5.66 MiB | 9.19 MiB | 14.85 MiB |
| 2,048 | 22.64 MiB | 9.19 MiB | 31.83 MiB |
| 8,192 | 90.56 MiB | 9.19 MiB | 99.75 MiB |
| 32,768 | 362.25 MiB | 9.19 MiB | 371.44 MiB |
| 131,072 | 1,449.00 MiB | 9.19 MiB | 1,458.19 MiB |

These numbers are runtime savings layered on top of PMRA's static weight-payload reduction.

## First Stack Test

The clean first experiment is model-forward evaluation, not GGUF runtime integration.

Evaluate four variants on the same public Wikitext validation prompts used by PMRA:

| Variant | Meaning |
|---|---|
| `q3_k_s` | uniform production control |
| `pmra` | PMRA knapsack weights only |
| `orbitquant` | fp/bf16 weights plus OrbitQuant activation policy |
| `pmra_orbitquant` | PMRA knapsack weights plus OrbitQuant activation policy |

Success criterion:

```text
PMRA+OrbitQuant keeps most of PMRA's NLL advantage over Q3_K_S
while reducing runtime KV/activation memory beyond the GGUF weight payload.
```

Implementation path:

1. Reuse PMRA's HF forward evaluator and `patch_all_from_source` / `apply_selection` path to materialize PMRA weights inside a live Transformers model.
2. Adapt OrbitQuant runtime patching to Gemma4 text modules:
   - locate text layers under `model.language_model.layers`
   - patch Gemma4 attention K/V after RoPE and before cache update
   - patch Gemma4 MLP intermediate after activation/product and before the output projection
3. Run a small calibration sweep on Gemma4 to choose layer sites and primitive choices under the PMRA weight state.
4. Compare against the PMRA-only model on disjoint held-out prompts.

## Conclusion

PMRA and OrbitQuant should stack architecturally. PMRA spends bytes across static GGUF weight tensors; OrbitQuant spends bits across runtime activation buses. The first real risk is interaction quality, not method overlap: PMRA's selected lower-bit weight tensors may shift which KV/MLP activation layers tolerate OrbitQuant best. That argues for rerunning the OrbitQuant allocator under the PMRA-patched Gemma4 model rather than copying SmolLM2 layer choices.
