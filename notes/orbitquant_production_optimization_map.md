# OrbitQuant Production Optimization Map

Date: 2026-05-17

## Goal

Turn OrbitQuant from an evaluator-grade runtime overlay into a production-grade compression policy that can broaden beyond the current safe3 Gemma4 result while staying measurable, stackable with PMRA, and implementable in a real inference path.

Current strongest Gemma4 point:

| Model state | NLL | Static payload | Runtime saving estimate |
|---|---:|---:|---:|
| PMRA | 12.818462 | 5.326613 bpw | n/a |
| PMRA + OrbitQuant safe3 | 12.834083 | 5.326613 bpw | 48.78 MiB |
| PMRA + OrbitQuant safe3 folded | 12.800727 | 5.326613 bpw | 48.78 MiB |

## Research Signals

The nearby literature points to four practical lessons:

1. Rotation is a real compression primitive. QuaRot applies computationally equivalent rotations across residual, feed-forward, attention, and KV surfaces to enable 4-bit inference. SpinQuant shows learned rotations can outperform arbitrary rotations.
2. KV cache production needs kernel-aware design. QServe reaches W4A8KV4 by pairing quantization choices with serving-system constraints. vLLM's FP8 KV-cache work emphasizes that long-context serving becomes memory-bound and that backend details decide whether quantized cache is fast.
3. KV keys and values deserve different treatment. KVQuant emphasizes per-channel key quantization, pre-RoPE key quantization, non-uniform datatypes, and outlier handling. RotateKV adds outlier-aware reordering, pre-RoPE grouped-head rotation, and attention-sink-aware protection.
4. Activation outliers can be moved, distributed, or clustered. SmoothQuant migrates activation difficulty into weights with an equivalent transform. RPTQ clusters channels by range. DuQuant combines rotation and permutation for outlier management.

## Highest-Value OrbitQuant Optimizations

### 1. Fold MLP Inverse Rotation Into `down_proj`

Current evaluator path:

```text
z -> R -> quantize -> R^T -> down_proj
```

Production path:

```text
z -> R -> quantize -> down_proj_rotated
```

For row-vector activations, current output is:

```text
q(zR) R^T W_down^T = q(zR) (W_down R)^T
```

So each selected MLP bus can precompute:

```text
W_down_rotated = W_down R
```

This removes the inverse activation rotation from runtime and makes the compressed MLP bus a candidate for a fused int2 activation x rotated-weight GEMM.

Test:

- Add a folded-weight MLP patch beside the current hook.
- Confirm exact equivalence before quantization and near-equivalence after quantization.
- Measure peak RAM and wall time on CPU for batch-1 decode and longer prefill.

Status:

- Implemented as `--mlp-fold-down-proj`.
- Float32 smoke test passes for identity, block-Hadamard, and Hadamard-plus prepermutation rotations.
- Gemma4 PMRA + folded safe3 validation on 128 Wikitext prompts landed at NLL `12.800727`, or `-0.017735` versus PMRA.

### 2. Split K and V Policies

Current Gemma4 hook quantizes K and V together with the same bits, rotation, and alpha. Literature and implementation reports repeatedly suggest K is more attention-score-sensitive while V is more MSE-like.

Candidate policies:

| Policy | K | V | Why |
|---|---|---|---|
| safe | 4-bit Hadamard | 3-bit Hadamard | protect attention scores |
| value-heavy | fp16/q8 K | 2/3-bit V | isolate value memory savings |
| key-corrected | 3-bit K + correction | 3-bit V | test TurboQuant/KVLinC-like residual correction |
| layer-gated | per-layer K/V bits | per-layer K/V bits | allocator chooses sensitivity |

Test:

- Extend `kv_layers` policy entries from layer-only to layer + `k_bits`, `v_bits`, `k_rotation`, `v_rotation`, `k_alpha`, `v_alpha`.
- Add variants: `pmra_k_only`, `pmra_v_only`, `pmra_kv_split`.
- Run Gemma4 Modal profiles with 128 prompts first, then long-context probes.

### 3. Pre-RoPE Key Rotation / Quantization

Current hook applies RoPE and then quantizes key states. KVQuant and RotateKV both point at pre-RoPE key handling as a serious optimization path.

Candidate:

```text
k_proj -> k_norm -> rotate/quantize/store pre-RoPE K -> apply RoPE at attention read
```

This is harder in a stock HF hook because cache storage and attention read are normally post-RoPE. It is worth prototyping in evaluator form first, then deciding whether the runtime target should be a custom cache class or backend integration.

Test:

- Implement evaluator-only pre-RoPE K quantization for non-shared KV layers.
- Compare attention score MSE and NLL against post-RoPE K quantization.
- Keep V path unchanged.

### 4. Attention Sink / Recent Token Protection

Low-bit cache often fails on rare high-impact tokens. A production policy can keep small token subsets higher precision:

- first N tokens
- sink tokens identified by calibration
- most recent window
- unusually high-norm K/V vectors

Test:

- Add `protected_prefix_tokens`, `protected_recent_tokens`, and `protected_norm_percentile`.
- Measure memory saved after protection overhead.
- Use long-context retrieval prompts, not only Wikitext NLL.

### 5. Non-Uniform Quantizers After Rotation

Current OrbitQuant uses uniform scalar quantization with per-vector absmax scaling and alpha clipping. TurboQuant suggests Lloyd-Max or distribution-matched scalar quantizers after rotation; KVQuant also supports non-uniform datatypes.

Candidate quantizers:

- symmetric uniform with learned alpha
- Gaussian Lloyd-Max centroids
- ternary plus sparse outlier side channel
- radius + normalized direction quantization for KV vectors

Test:

- Implement a pluggable quantizer interface for `quantize_last_dim`.
- Start with Gaussian Lloyd-Max codebooks for 2, 3, and 4 bits.
- Compare reconstruction MSE, attention-score MSE, and NLL.

### 6. Diagonal Smoothing Before Rotation

SmoothQuant-style exact diagonal scaling can reduce activation range before quantization:

```text
z -> D -> R -> quantize -> W_down D^-1 R folded into boundary
```

For MLP buses, this can be folded into `down_proj` alongside the rotation. This combines the best parts of smoothing and Hadamard-plus pre-staging.

Test:

- Calibrate per-channel activation max/RMS.
- Sweep diagonal smoothing strength before Hadamard-plus.
- Fold the inverse into `W_down_rotated`.

### 7. Allocator Upgrade: Device-Aware Objective

The existing allocator optimizes quality metrics and estimated memory. Production needs a score that includes expected runtime cost:

```text
score = NLL_delta + lambda_mem * memory_cost + lambda_time * transform_cost + lambda_risk * interaction_risk
```

Device profiles:

- laptop CPU: minimize overhead and peak RAM
- CUDA serving: optimize fused kernels and KV bandwidth
- vLLM-style serving: respect paged cache and attention backend constraints

Test:

- Add policy cost fields: `rotation_flops`, `extra_weight_bytes`, `metadata_bytes`, `cache_layout`.
- Rank policies separately for CPU-local and server profiles.

## Production Path

### First Runtime Target

The lowest-risk production step is a folded MLP policy plus split K/V evaluator:

1. Implement folded MLP rotation.
2. Implement split K/V policy.
3. Run Gemma4 PMRA + OrbitQuant on Modal with 128 prompts.
4. Run laptop CPU smoke benchmarks: peak RAM, tokens/sec, and output sanity.
5. Export `compression_config.json` with explicit runtime requirements.

### Backend Target

For a local-first prototype, keep the HF/Transformers runtime and use it as a correctness harness. For broader deployment, the likely backend targets are:

- llama.cpp/GGML for local CPU/GPU users and GGUF-style deployment.
- vLLM for server-side paged KV cache and attention kernels.
- a small standalone C/C++ reference kernel for the MLP folded bus if llama.cpp integration is too heavy at first.

## Near-Term Experiment Queue

| Priority | Experiment | Expected value |
|---:|---|---|
| 1 | Fold inverse MLP rotation into `down_proj` | turns MLP bus into a production-shaped primitive |
| 2 | K-only / V-only / split K/V policies | broadens KV safely and reveals sensitivity |
| 3 | Gaussian/Lloyd-Max scalar codebooks | likely improves 3-bit KV and 2-bit MLP quality |
| 4 | diagonal smoothing + Hadamard-plus | could rescue more MLP layers |
| 5 | attention sink and recent-token protection | makes low-bit KV safer for long context |
| 6 | pre-RoPE key quantization | higher-risk, potentially large KV gain |

## References To Track

- QuaRot: https://arxiv.org/abs/2404.00456
- SpinQuant: https://arxiv.org/abs/2405.16406
- QServe: https://arxiv.org/abs/2405.04532
- TurboQuant: https://huggingface.co/papers/2504.19874
- SmoothQuant: https://arxiv.org/abs/2211.10438
- RPTQ: https://arxiv.org/abs/2304.01089
- DuQuant: https://huggingface.co/papers/2406.01721
- KVQuant: https://arxiv.org/abs/2401.18079
- RotateKV: https://arxiv.org/abs/2501.16383
- vLLM FP8 KV-cache note: https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-22-fp8-kvcache.md
