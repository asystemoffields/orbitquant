# SmolLM3 Orbit Probe Log

Date: 2026-05-13

Model: `HuggingFaceTB/SmolLM3-3B`

Local setup:

- CPU-only PyTorch
- SmolLM3 weight access through safetensor shards
- Layer-local probes using real model weights and synthetic activations

## Probe 1: Value/O Attention Basis Rotation

Target: layer 0, GQA value heads.

Exact symmetry:

```text
V_head' = R^T V_head
O_segment' = O_segment R
```

For each KV head, the same rotation is applied to the `v_proj` output subspace and to every `o_proj` input segment consumed by the query heads sharing that KV head.

At full precision the transformed operator matches the original at numerical noise levels around `1e-13` relative MSE.

### Layer 0, KV Head 0

| Bits | Identity rel MSE | Best rotated rel MSE | Best candidate |
|---:|---:|---:|---|
| 4 | 0.028082 | 0.020417 | random orthogonal |
| 3 | 0.154945 | 0.113159 | Hadamard + sign/perm |
| 2 | 1.106317 | 0.980574 | random orthogonal |

### Layer 0, All KV Heads at 3-Bit

| KV head | Identity rel MSE | Best rotated rel MSE | Best candidate |
|---:|---:|---:|---|
| 0 | 0.154945 | 0.113159 | Hadamard + sign/perm |
| 1 | 0.129494 | 0.123748 | random orthogonal |
| 2 | 0.166354 | 0.131157 | Hadamard |
| 3 | 0.201199 | 0.162043 | Hadamard + sign/perm |

Observation: value/O basis choice matters. The strongest practical candidate is Hadamard-style rotation because it has cheap structured metadata and fast kernels are plausible.

## Probe 2: MLP Up/Down Scaling

Target: layer 0 SwiGLU `up_proj` and `down_proj`.

Exact symmetry:

```text
up_i' = s_i up_i
down_col_i' = down_col_i / s_i
```

The gate projection is held fixed. At full precision the transformed MLP path matches the original at numerical noise levels around `1e-13` relative MSE.

### Layer 0

| Bits | Identity rel MSE | Best transformed rel MSE | Best candidate |
|---:|---:|---:|---|
| 3 | 0.116249 | 0.115956 | norm balance |
| 2 | 0.886150 | 0.886150 | identity |

Observation: simple norm or max balancing is weak for this MLP path. The scaling orbit likely needs an activation-aware objective, especially because group quantization and the fixed gate path make naive channel balancing too blunt.

## Next Bets

1. Replace synthetic activations with captured activations from real prompts.
2. Search Hadamard/sign/permutation choices across all layers and KV heads.
3. Add an activation-weighted objective for MLP scaling.
4. Test group size sensitivity, especially `32`, `64`, and `128`.
5. Add a small low-rank correction map after the rotated low-bit base.

## Probe 3: Full-Model Smol135 Logit Drift

Model: `HuggingFaceTB/SmolLM2-135M`

Script: `scripts/smol135_full_probe.py`

This probe loads the full 135M model on CPU, records baseline logits for eight short prompts, applies one compressed layer-local edit, and measures next-token loss, logit KL from baseline, logit MSE, and top-10 overlap.

Baseline on the prompt set:

| Loss | Perplexity |
|---:|---:|
| 4.119296 | 61.515896 |

### Value/O Hadamard Rotation, Layer 0, 3-Bit

| KV head | Candidate | Loss | KL from baseline | Logit MSE | Top-10 overlap |
|---:|---|---:|---:|---:|---:|
| 0 | identity 3-bit | 4.081020 | 0.006607 | 0.066180 | 0.957143 |
| 0 | Hadamard 3-bit | 4.133381 | 0.004380 | 0.120443 | 0.961905 |
| 1 | identity 3-bit | 4.116193 | 0.009533 | 0.141795 | 0.959524 |
| 1 | Hadamard 3-bit | 4.122557 | 0.007112 | 0.104366 | 0.964286 |
| 2 | identity 3-bit | 4.185207 | 0.024608 | 0.636594 | 0.917857 |
| 2 | Hadamard 3-bit | 4.127299 | 0.018203 | 0.258733 | 0.935714 |

Observation: Hadamard value/O rotation reduces KL drift and improves top-10 overlap for all three KV heads in this small full-model probe. Head 2 shows the largest improvement.

### Typical-Subspace V Projection, Layer 0

Uniform 3-bit `v_proj` quantization:

| Loss | KL from baseline | Logit MSE | Top-10 overlap |
|---:|---:|---:|---:|
| 4.135268 | 0.025581 | 0.481513 | 0.919048 |

PCA-typical mixed precision over input directions:

| Top fraction | Effective bits | Retained activation energy | Loss | KL from baseline | Logit MSE | Top-10 overlap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 2.500000 | 1.000000 | 4.130467 | 0.005917 | 0.211771 | 0.945238 |
| 0.10 | 2.201389 | 0.979007 | 4.126006 | 0.007750 | 0.260847 | 0.946429 |

Observation: preserving high-precision capacity for the activation-typical directions can beat uniform 3-bit quantization on behavioral drift, even with a lower average bit budget for the input-direction representation. This is a first concrete version of the "typical subspace" idea.

## Probe 4: Held-Out Sweeps

Script: `scripts/smol135_sweep.py`

The first full-model probe reused the same prompt set for activation capture and evaluation. The sweep script splits calibration prompts from held-out eval prompts.

### Typical-Subspace V Projection on Held-Out Layers

Layers tested: `0,1,2`

Top fractions tested: `0.05,0.10`

Summary:

| Metric | Result |
|---|---:|
| KL wins vs uniform 3-bit | 2 / 6 |
| Logit MSE wins vs uniform 3-bit | 2 / 6 |
| Top-10 overlap wins vs uniform 3-bit | 2 / 6 |
| Mean KL delta | +0.162968 |
| Mean logit MSE delta | +2.259408 |
| Mean top-10 overlap delta | -0.104889 |

Observation: the naive PCA-typical `v_proj` edit was a layer-0 effect and did not generalize to early held-out layers. This idea needs a better boundary objective or a different place to act.

### Value/O Weight Orbit, Calibration-Selected Rotation

Layers tested: `0,1,2,4,8,12,16,20,24,29`

Rotations tested: `identity`, `hadamard`, `hadamard_sign_perm`, `random_orthogonal`

Selection rule: choose the rotation by calibration KL, then report held-out eval drift.

At 2-bit:

| Metric | Result |
|---|---:|
| KL wins vs identity | 18 / 30 |
| Logit MSE wins vs identity | 14 / 30 |
| Top-10 overlap wins vs identity | 13 / 30 |
| Mean KL delta | -0.052255 |
| Mean logit MSE delta | -1.006889 |
| Mean top-10 overlap delta | +0.014289 |

Best calibration-selected rotations:

| Rotation | Count |
|---|---:|
| identity | 11 |
| hadamard | 5 |
| hadamard + sign/perm | 6 |
| random orthogonal | 8 |

Observation: static value/O orbit search is useful but selective. The real pattern is not "Hadamard everywhere"; it is "some heads have fragile value/O coordinates, and a tiny orbit search can save them at very low bit width."

## Probe 5: TurboQuant-Style KV Activation Rotations

Script: `scripts/smol135_kv_rotation_probe.py`

This probe directly tests the TurboQuant-like idea on captured Smol135 KV activations. For each layer and KV head, it quantizes K/V vectors in different bases and measures:

- key vector relative MSE
- value vector relative MSE
- query-key attention score relative MSE

Layers tested: all 30 layers.

Rotations tested: `identity`, `hadamard`, `random_orthogonal`.

### 3-Bit KV Activation Quantization

| Metric | Result |
|---|---:|
| Groups | 90 |
| Identity best | 0 |
| Hadamard best | 63 |
| Random orthogonal best | 27 |
| Mean best key-score delta vs identity | -0.249831 |
| Mean best key-vector delta vs identity | -0.123563 |
| Mean best value-vector delta vs identity | -0.033402 |

### 2-Bit KV Activation Quantization

| Metric | Result |
|---|---:|
| Groups | 90 |
| Identity best | 13 |
| Hadamard best | 49 |
| Random orthogonal best | 28 |
| Mean best key-score delta vs identity | -0.655841 |
| Mean best key-vector delta vs identity | -0.024182 |
| Mean best value-vector delta vs identity | -0.032973 |

Observation: this is the strongest signal so far. Rotating KV activations before scalar quantization consistently reduces attention score distortion across the whole 135M model. This lines up with TurboQuant's rotation-based KV-cache result and explains why the value/O weight orbit experiment worked in some heads: both are fighting coordinate-local outliers before scalar quantization.

## Probe 6: MLP Intermediate Activation Rotations

Script: `scripts/smol135_mlp_activation_rotation_probe.py`

This tests a TurboQuant-like operation outside the KV cache. For each MLP layer:

```text
z = SiLU(gate_proj(x)) * up_proj(x)
y = down_proj(z)
```

The probe rotates and scalar-quantizes `z`, rotates it back, then measures:

- intermediate activation relative MSE
- `down_proj` boundary-output relative MSE

The boundary-output metric is the important one because it asks whether the next model-visible map is preserved.

### 3-Bit MLP Intermediate Quantization

Layers tested: all 30 layers.

Rotations tested: `identity`, `block_hadamard`, `block_hadamard_sign_perm`.

Block size: `512`, giving three independent Hadamard blocks across the 1536-wide MLP intermediate dimension.

| Metric | Result |
|---|---:|
| Groups | 30 |
| Identity best | 0 |
| Block Hadamard best | 24 |
| Block Hadamard + sign/perm best | 6 |
| Mean best activation delta vs identity | -0.221997 |
| Mean best down-output delta vs identity | -0.209550 |

### 2-Bit MLP Intermediate Quantization

Layers tested: `0,1,2,4,8,12,16,20,24,29`.

| Metric | Result |
|---|---:|
| Groups | 10 |
| Identity best | 4 |
| Block Hadamard best | 5 |
| Block Hadamard + sign/perm best | 1 |
| Mean best activation delta vs identity | +0.048146 |
| Mean best down-output delta vs identity | -0.049612 |

Observation: the TurboQuant pattern transfers beyond KV cache. At 3-bit, rotating the MLP intermediate activation before scalar quantization improves every layer tested. At 2-bit, the effect is still positive on down-projection boundary error but less uniform. This looks stackable with KV-cache rotation because it acts on a different runtime activation bus.

## Current Stack Hypothesis

The emerging stack is:

```text
1. KV cache: rotate K/V vectors before low-bit cache storage.
2. Attention value/output weights: selectively search exact value/O head-space orbits before static low-bit storage.
3. MLP intermediate activations: rotate SwiGLU product vectors before low-bit activation storage or low-bit down-projection input.
4. Optional correction: add a tiny residual or QJL-style estimator where inner products or boundary maps show systematic bias.
```

The real lesson from TurboQuant is not just "rotate KV." It is: when a high-dimensional vector is about to suffer coordinate-wise scalar quantization, first make the coordinate distribution predictable and outlier-resistant, then preserve the boundary operation that consumes that vector.

## Probe 7: Calibrated 2-Bit MLP Ternary Scale Search

Script: `scripts/smol135_mlp_ternary_search.py`

This probe targets the weakness found in Probe 6: at 2-bit, absmax ternary quantization made the threshold too coarse after rotation. The new search sweeps clipping-scale multipliers:

```text
alpha in {1.0, 0.75, 0.5, 0.375, 0.25}
```

The chosen rotation/alpha is selected on calibration prompts and evaluated on held-out prompts.

Layers tested: all 30 layers.

Bits: `2`

Rotations tested: `identity`, `block_hadamard`, `block_hadamard_sign_perm`.

Block size: `512`.

### Full 30-Layer Result

| Metric | Result |
|---|---:|
| Groups | 30 |
| Down-output wins vs identity absmax ternary | 30 / 30 |
| Activation wins vs identity absmax ternary | 26 / 30 |
| Mean eval activation delta vs identity | -0.393494 |
| Mean eval down-output delta vs identity | -0.381244 |

Best calibration-selected candidates:

| Candidate | Count |
|---|---:|
| block Hadamard @ 0.375 | 18 |
| block Hadamard + sign/perm @ 0.375 | 4 |
| block Hadamard @ 1.0 | 3 |
| block Hadamard @ 0.5 | 2 |
| identity @ 0.75 | 1 |
| block Hadamard + sign/perm @ 1.0 | 1 |
| block Hadamard + sign/perm @ 0.75 | 1 |

Observation: this is the strongest non-KV result so far. The 2-bit MLP intermediate bus becomes reliably compressible once rotation is paired with calibrated ternary clipping. The common winning value, `alpha=0.375`, confirms the failure mode: absmax scaling was wasting the ternary codebook on rare extremes. Clipping plus block-Hadamard turns the 2-bit representation into a usable boundary-preserving code.

## Probe 8: End-to-End 2-Bit MLP Stack and Layer Allocator

Scripts:

- `scripts/smol135_mlp_stack_probe.py`
- `scripts/smol135_mlp_impact_sweep.py`

Probe 7 validated the local MLP boundary. Probe 8 asks whether the same scheme survives a real model forward pass after patching MLP modules.

The patched MLP forward is:

```text
z = SiLU(gate_proj(x)) * up_proj(x)
z_q = dequantize_2bit(round((R z) / scale)) R^T
out = down_proj(z_q)
```

The rotation and clipping scale are chosen per layer on calibration prompts. Metrics below are held-out eval prompt metrics against the unmodified model logits.

### Ten-Layer Stack

Layers: `16,17,12,6,14,21,5,19`

| Candidate | Loss | KL from baseline | Top-10 overlap | Loss delta |
|---|---:|---:|---:|---:|
| identity absmax ternary | 5.270434 | 0.603182 | 0.620667 | +0.592399 |
| identity clipped ternary @ 0.375 | 4.952206 | 0.356574 | 0.714000 | +0.274171 |
| calibrated rotation ternary | 4.782205 | 0.101035 | 0.838000 | +0.104170 |

### Twelve-Layer Stack

Layers: `16,17,12,6,14,21,5,19,25,4,26,13`

| Candidate | Loss | KL from baseline | Top-10 overlap | Loss delta |
|---|---:|---:|---:|---:|
| identity absmax ternary | 5.784342 | 0.924803 | 0.538000 | +1.106307 |
| identity clipped ternary @ 0.375 | 5.557739 | 0.618223 | 0.640667 | +0.879704 |
| calibrated rotation ternary | 4.959572 | 0.167981 | 0.804667 | +0.281538 |

### Per-Layer Allocator Sweep

Each layer was individually patched and scored end-to-end. Then layers were stacked by lowest calibrated single-layer KL.

Best individual layers by calibrated KL:

| Rank | Layer | KL | Top-10 overlap |
|---:|---:|---:|---:|
| 1 | 16 | 0.009947 | 0.949333 |
| 2 | 17 | 0.010640 | 0.950000 |
| 3 | 12 | 0.011449 | 0.947333 |
| 4 | 6 | 0.011637 | 0.940667 |
| 5 | 14 | 0.012142 | 0.940000 |
| 6 | 21 | 0.013144 | 0.931333 |
| 7 | 5 | 0.013189 | 0.943333 |
| 8 | 19 | 0.013405 | 0.942667 |

Stacking by best individual layers:

| Layers compressed | KL | Top-10 overlap | Loss delta |
|---:|---:|---:|---:|
| 4 | 0.043243 | 0.898000 | -0.007729 |
| 8 | 0.101035 | 0.838000 | +0.104170 |
| 12 | 0.167981 | 0.804667 | +0.281538 |
| 16 | 0.275144 | 0.750667 | +0.350428 |
| 20 | 0.385542 | 0.708000 | +0.481235 |

Observation: full-model behavior confirms an allocator-shaped result. Aggressively ternarizing every MLP bus is too much, but selected layers can use calibrated 2-bit rotated MLP activations with modest held-out logit drift. The 8-layer stack is the cleanest current tradeoff: eight MLP intermediate buses at 2-bit, KL `0.101`, top-10 overlap `0.838`, and only `+0.104` loss delta.

### Replication on Original SmolLM-135M

Model: `HuggingFaceTB/SmolLM-135M`

Script: `scripts/smol135_mlp_impact_sweep.py`

Same method: per-layer calibrated 2-bit MLP intermediate rotation/scale, then stack by lowest individual held-out KL.

Best individual layers:

| Rank | Layer | KL | Top-10 overlap |
|---:|---:|---:|---:|
| 1 | 6 | 0.008285 | 0.943333 |
| 2 | 15 | 0.009515 | 0.938667 |
| 3 | 16 | 0.010194 | 0.935333 |
| 4 | 17 | 0.010322 | 0.940667 |
| 5 | 5 | 0.010898 | 0.933333 |
| 6 | 14 | 0.011197 | 0.936667 |
| 7 | 13 | 0.012309 | 0.940667 |
| 8 | 22 | 0.013024 | 0.929333 |

Stacking by best individual layers:

| Layers compressed | KL | Top-10 overlap | Loss delta |
|---:|---:|---:|---:|
| 4 | 0.039959 | 0.876000 | +0.054014 |
| 8 | 0.095319 | 0.823333 | +0.056773 |
| 12 | 0.160971 | 0.780667 | +0.134200 |

Observation: the result replicates on the earlier SmolLM-135M checkpoint. The exact layer ranking shifts, but the pattern is stable: a calibrated rotated 2-bit MLP intermediate bus can be allocated to a meaningful subset of layers with low end-to-end logit drift.

## Probe 9: Fused KV + MLP Activation Stack

Scripts:

- `scripts/smol135_kv_impact_sweep.py`
- `scripts/smol135_fused_stack_probe.py`

This probe stacks two independent TurboQuant-like runtime buses:

```text
KV attention bus: K/V activations -> Hadamard -> 3-bit scalar quantization -> attention
MLP bus: SwiGLU intermediate z -> block Hadamard -> calibrated 2-bit ternary -> down_proj
```

### KV Layer Allocator

Model: `HuggingFaceTB/SmolLM2-135M`

Bits: `3`

Rotations: `identity`, `hadamard`

Selection: choose rotation by calibration KL, rank layers by held-out KL.

Best KV layers:

| Rank | Layer | Rotation | KL | Top-10 overlap |
|---:|---:|---|---:|---:|
| 1 | 28 | hadamard | 0.002544 | 0.969333 |
| 2 | 8 | hadamard | 0.008117 | 0.952667 |
| 3 | 24 | hadamard | 0.009184 | 0.952000 |
| 4 | 15 | hadamard | 0.009720 | 0.948000 |
| 5 | 14 | hadamard | 0.009952 | 0.951333 |
| 6 | 13 | hadamard | 0.010545 | 0.948000 |
| 7 | 16 | hadamard | 0.011326 | 0.938000 |
| 8 | 27 | hadamard | 0.012480 | 0.943333 |

KV-only stacks:

| KV layers compressed | KL | Top-10 overlap | Loss delta |
|---:|---:|---:|---:|
| 4 | 0.032707 | 0.898667 | -0.033176 |
| 8 | 0.099218 | 0.831333 | -0.242401 |
| 12 | 0.201036 | 0.784000 | -0.389381 |
| 16 | 0.296192 | 0.734667 | -0.550792 |

### Fused Stack Results

Baseline eval loss: `4.678035`.

Compact fused stack:

- KV layers: `28,8,24,15`
- KV scheme: `3-bit`, Hadamard, `alpha=0.75`
- MLP layers: `16,17,12,6`
- MLP scheme: calibrated rotated `2-bit`

| Candidate | KL | Top-10 overlap | Loss delta |
|---|---:|---:|---:|
| KV only, Hadamard 3-bit | 0.028942 | 0.900667 | -0.056113 |
| MLP only, calibrated rotated 2-bit | 0.043243 | 0.898000 | -0.007729 |
| fused calibrated stack | 0.065610 | 0.862667 | -0.038112 |
| fused naive identity absmax stack | 0.323074 | 0.711333 | +0.134828 |

Same compact stack with KV `alpha=1.0`:

| Candidate | KL | Top-10 overlap | Loss delta |
|---|---:|---:|---:|
| fused calibrated stack | 0.068527 | 0.866667 | -0.051996 |
| fused naive identity absmax stack | 0.304321 | 0.730667 | +0.181448 |

Larger fused stacks:

| KV layers | MLP layers | KL | Top-10 overlap | Loss delta |
|---:|---:|---:|---:|---:|
| 4 | 8 | 0.129288 | 0.796000 | -0.007904 |
| 8 | 4 | 0.134726 | 0.800000 | -0.222329 |
| 8 | 8 | 0.196368 | 0.762000 | -0.113562 |

Observation: the two compression methods stack. Drift is not perfectly additive, so an allocator is necessary, but the compact stack is a real composed result: selected KV layers at 3-bit plus selected MLP buses at 2-bit with KL `0.0656`, top-10 overlap `0.8627`, and no loss increase on the held-out prompt set. The naive fused stack is roughly five times worse by KL.

### Broad 78-Prompt Validation

Prompt files:

- `prompts/broad_calibration.txt`
- `prompts/broad_eval.txt`

The broad split uses 78 calibration prompts and 78 held-out eval prompts. This is the current best validation set for the small CPU probes.

Best broad KV layers by individual held-out KL:

| Rank | Layer | Rotation | KL | Top-10 overlap |
|---:|---:|---|---:|---:|
| 1 | 28 | hadamard | 0.002539 | 0.968732 |
| 2 | 24 | hadamard | 0.006191 | 0.961408 |
| 3 | 26 | hadamard | 0.007856 | 0.950423 |
| 4 | 14 | hadamard | 0.008163 | 0.944930 |
| 5 | 17 | hadamard | 0.009513 | 0.950845 |
| 6 | 15 | hadamard | 0.009671 | 0.945352 |
| 7 | 13 | hadamard | 0.009795 | 0.946479 |
| 8 | 25 | hadamard | 0.010619 | 0.947042 |

Broad KV-only stacks:

| KV layers compressed | Layers | KL | Top-10 overlap | Loss delta |
|---:|---|---:|---:|---:|
| 4 | `28,24,26,14` | 0.026430 | 0.904366 | -0.046800 |
| 8 | `28,24,26,14,17,15,13,25` | 0.078045 | 0.857465 | -0.077951 |
| 12 | top 12 | 0.125706 | 0.816901 | -0.041762 |
| 16 | top 16 | 0.211020 | 0.763944 | -0.205419 |

Best broad MLP layers by individual held-out KL:

| Rank | Layer | KL | Top-10 overlap |
|---:|---:|---:|---:|
| 1 | 16 | 0.005924 | 0.964085 |
| 2 | 17 | 0.007054 | 0.956901 |
| 3 | 15 | 0.008063 | 0.951690 |
| 4 | 12 | 0.008108 | 0.954789 |
| 5 | 13 | 0.008843 | 0.952535 |
| 6 | 14 | 0.009023 | 0.947606 |
| 7 | 5 | 0.009340 | 0.949718 |
| 8 | 19 | 0.010590 | 0.948592 |

Broad MLP-only stacks:

| MLP layers compressed | Layers | KL | Top-10 overlap | Loss delta |
|---:|---|---:|---:|---:|
| 4 | `16,17,15,12` | 0.031435 | 0.904366 | +0.019965 |
| 8 | `16,17,15,12,13,14,5,19` | 0.075496 | 0.850986 | +0.098282 |
| 12 | top 12 | 0.135762 | 0.813944 | +0.176020 |

Broad fused stacks:

| KV layers | MLP layers | KL | Top-10 overlap | Loss delta |
|---:|---:|---:|---:|---:|
| 4 | 4 | 0.059405 | 0.858310 | -0.033383 |
| 4 | 8 | 0.100472 | 0.816620 | +0.052434 |
| 8 | 4 | 0.106520 | 0.818732 | -0.077473 |
| 8 | 8 | 0.144067 | 0.785070 | -0.052279 |

Observation: the broad results keep the central pattern intact. A compact fused stack can compress four KV layers to 3-bit and four MLP intermediate buses to 2-bit with KL `0.0594`, top-10 overlap `0.8583`, and a small loss decrease on the broad held-out set. The naive fused identity/absmax counterpart for the same broad 4+4 stack has KL `0.246606`, so the rotation/calibration stack is roughly four times better by KL.

## Probe 10: Joint Fused Allocator

Script: `scripts/smol135_joint_allocator_probe.py`

This probe starts from the broad compact fused stack:

```text
KV: 28,24,26,14 at 3-bit Hadamard
MLP: 16,17,15,12 at calibrated rotated 2-bit
```

Then it tries one extra KV or MLP layer at a time and measures the actual fused held-out KL after adding that layer.

### One-Step Marginal Results

Base stack: KL `0.059405`, top-10 overlap `0.858310`, loss delta `-0.033383`.

Best additions by actual fused marginal KL:

| Rank | Add | Independent KL | New stack KL | Marginal KL |
|---:|---|---:|---:|---:|
| 1 | KV layer 9 | 0.012327 | 0.067976 | +0.008571 |
| 2 | KV layer 13 | 0.009795 | 0.068425 | +0.009020 |
| 3 | MLP layer 14 | 0.009023 | 0.068941 | +0.009536 |
| 4 | MLP layer 13 | 0.008843 | 0.069244 | +0.009840 |
| 5 | KV layer 17 | 0.009513 | 0.069246 | +0.009841 |

Largest warning case:

| Add | Independent KL | New stack KL | Marginal KL |
|---|---:|---:|---:|
| KV layer 4 | 0.012321 | 0.094465 | +0.035060 |

Observation: independent layer ranking is a useful first pass, but it misses real fused interactions. KV layer 9 is only middling by independent KL and is the best next fused addition. KV layer 4 looks acceptable alone and is the worst one-step addition in this candidate pool.

### Four-Step Greedy Allocator

Candidate pool: next eight KV layers and next eight MLP layers after the compact base, from the broad impact sweeps.

Greedy choices:

| Step | Added bus | Stack KL | Top-10 overlap | Current marginal KL |
|---:|---|---:|---:|---:|
| 1 | KV layer 9 | 0.067976 | 0.856197 | +0.008571 |
| 2 | MLP layer 18 | 0.077253 | 0.847042 | +0.009276 |
| 3 | MLP layer 6 | 0.088143 | 0.826761 | +0.010890 |
| 4 | MLP layer 19 | 0.098357 | 0.820704 | +0.010214 |

Comparison at 12 total compressed buses:

| Allocator | KV buses | MLP buses | KL | Top-10 overlap | Loss delta |
|---|---:|---:|---:|---:|---:|
| independent top KV + top MLP | 4 | 8 | 0.100472 | 0.816620 | +0.052434 |
| independent top KV + top MLP | 8 | 4 | 0.106520 | 0.818732 | -0.077473 |
| greedy joint allocator | 5 | 7 | 0.098357 | 0.820704 | +0.027793 |

Observation: the first greedy allocator improves the current 12-bus frontier slightly and, more importantly, confirms that the allocator should be interaction-aware. The compressed stack is acting like a coupled system: the best next bus depends on what has already been compressed.

## Sidecar Metaphor: Touring Stage Adaptation

Working metaphor: an LLM is a vast novel performed live, token by token, by a theater company. Compression is making that performance fit into a smaller touring production: fewer trunks, smaller cue sheets, simpler props, and tighter staging, while the audience still recognizes the same plot, callbacks, tone, and timing.

| Technical concept | Stage-adaptation term |
|---|---|
| Weights | Script, stage directions, character notes, and learned habits of the company |
| Weight quantization | Rewriting stage instructions with fewer exact shades |
| Activations | What is happening in the actors' minds and on stage during a line |
| Activation compression | Recording the live state with fewer marks on the cue sheet |
| Basis rotation / isomorphism | Restaging the same dramatic information across actors, props, and cues |
| Outlier channels | A performer or prop carrying too many important meanings at once |
| Rotation before quantization | Redistributing meaning so fewer single cues need impossible precision |
| KV cache | The running prompt-book of prior names, promises, and callbacks |
| MLP intermediate activations | Backstage transformations between spoken lines |
| Memory allocation | Deciding how much trunk space goes to scripts, prompt-books, props, and backstage machinery |
| Behavioral fidelity | The audience still sees the same play |

In this metaphor, the TurboQuant-style result says: the original staging made a few cues carry too much meaning. Rotation restages the same play so the meaning is spread across the company. After that, smaller cue sheets survive better because fewer individual cues have to preserve an entire scene by themselves.

## Probe 11: Hadamard-Plus MLP Bus Search

Scripts:

- `scripts/smol135_mlp_hadamard_plus_probe.py`
- `scripts/smol135_mlp_hadamard_plus_impact_sweep.py`
- `scripts/smol135_fused_hadamard_plus_probe.py`

### Hypothesis

Hadamard is already extremal for uniform spreading. For any real orthogonal matrix `R`, the largest absolute entry must satisfy:

```text
max_ij |R_ij| >= 1 / sqrt(d)
```

A Hadamard matrix hits this lower bound exactly, with every entry equal to `+/- 1/sqrt(d)`. So if the target is generic, model-agnostic coordinate spreading, Hadamard is already the clean universal move.

The MLP bus has extra structure:

```text
z = SiLU(gate_proj(x)) * up_proj(x)
y = down_proj(z)
```

The metric we care about is not just whether `z` survives quantization. It is whether `down_proj(z)` survives after quantizing a transformed `z`. That makes the next boundary map part of the compression problem.

The "more Hadamard than Hadamard" idea is therefore:

```text
keep Hadamard's cheap uniform spreading,
but choose the staging before the block Hadamard using activation and down_proj structure.
```

Plain block-Hadamard spreads whatever channels happen to be in the same block. A smarter pre-permutation can distribute high-energy or high-impact channels across blocks before the Hadamard mixing step. This keeps the fast kernel shape but makes the block assignment less accidental.

### Candidates

The prepared probe compares:

| Candidate | Idea |
|---|---|
| `identity` | 2-bit ternary baseline |
| `block_hadamard` | current known primitive |
| `block_hadamard_sign` | cheap sign-randomized variant |
| `preperm_activation_rms_hadamard` | distribute high-RMS activation channels across blocks |
| `preperm_activation_max_hadamard` | distribute rare activation spikes across blocks |
| `preperm_down_norm_hadamard` | distribute channels with large `down_proj` column norms |
| `preperm_boundary_rms_hadamard` | distribute channels scored by activation RMS times `down_proj` norm |
| `preperm_boundary_max_hadamard` | distribute spike-sensitive channels scored by activation max times `down_proj` norm |
| `interleave_boundary_rms_hadamard` | deterministic interleaving version of boundary-aware spreading |
| `random_preperm_hadamard` | random pre-permutation trials for comparison |

Each candidate is paired with the existing ternary clipping sweep:

```text
alpha in {1.0, 0.75, 0.5, 0.375, 0.25}
```

### Success Criteria

Primary metric:

```text
eval down_output_rel_mse vs best plain block_hadamard
```

The first signal is meaningful if the best Hadamard-plus candidate:

- wins over plain block-Hadamard on a majority of tested layers,
- improves mean held-out `down_output_rel_mse`,
- does not win only by overfitting calibration layers or prompts.

If this local boundary test wins, the next step is to add the winning candidate family to the end-to-end MLP impact sweep and then back into the fused beam allocator.

### Ready Command

```powershell
$layers = (0..29) -join ','
python scripts/smol135_mlp_hadamard_plus_probe.py --layers $layers --bits 2 --alphas 1.0,0.75,0.5,0.375,0.25 --block-size 512 --calibration-prompts prompts\broad_calibration.txt --eval-prompts prompts\broad_eval.txt --max-length 64
```

### Local Boundary Result

Result: `results/smol135_mlp_hadamard_plus_2bit_20260513_131208.json`

All 30 layers were tested on the broad calibration/eval split. The candidate for each layer was selected by calibration `down_output_rel_mse` and reported on held-out eval prompts.

| Metric | Result |
|---|---:|
| Layers tested | 30 |
| Wins vs best plain block-Hadamard | 29 / 30 |
| Wins vs identity absmax | 30 / 30 |
| Mean eval down-output delta vs plain | -0.012068 |
| Mean eval activation delta vs plain | -0.010474 |

Best candidate counts:

| Candidate | Count |
|---|---:|
| `preperm_activation_rms_hadamard@0.375` | 8 |
| `preperm_activation_max_hadamard@0.375` | 5 |
| `preperm_boundary_rms_hadamard@0.375` | 3 |
| `preperm_activation_rms_hadamard@0.5` | 2 |
| `preperm_down_norm_hadamard@1.0` | 2 |
| `preperm_boundary_max_hadamard@0.375` | 2 |
| `interleave_boundary_rms_hadamard@0.375` | 2 |

Observation: locally, the idea is real. Plain Hadamard is already the best universal spreader, but activation/down-proj-aware pre-staging improves the 2-bit MLP boundary almost everywhere.

### End-to-End MLP Impact

Result: `results/smol135_mlp_hadamard_plus_impact_2bit_20260513_132208.json`

The locally selected Hadamard-plus candidate was patched into the full model and compared to the locally best plain block-Hadamard candidate.

| Metric | Result |
|---|---:|
| Single-layer KL wins vs plain | 17 / 30 |
| Mean single-layer KL delta vs plain | -0.037109 |
| Mean single-layer top-10 delta vs plain | +0.007385 |

Stack results:

| MLP layers compressed | Hadamard-plus KL | Plain KL | Delta | Plus top-10 | Plain top-10 |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.030067 | 0.031435 | -0.001369 | 0.906197 | 0.904366 |
| 8 | 0.078660 | 0.075501 | +0.003159 | 0.850563 | 0.850986 |
| 12 | 0.133025 | 0.136729 | -0.003704 | 0.815493 | 0.813239 |

Observation: the local boundary win partially transfers. It is a small end-to-end improvement at 4 and 12 MLP buses, but not at 8. The full model is sensitive to layer selection and interactions.

### Fused KV + MLP Checks

Fused compact same-layer check:

| Stack | KL | Top-10 overlap | Loss delta |
|---|---:|---:|---:|
| KV4 + MLP4 Hadamard-plus, MLP layers `17,16,15,13` | 0.059490 | 0.857465 | -0.032892 |
| KV4 + same MLP4 plain Hadamard | 0.063148 | 0.854507 | -0.018344 |
| KV4 + original top MLP4 plain Hadamard `16,17,15,12` | 0.059405 | 0.858310 | -0.033383 |

Fused larger same-layer check:

| Stack | KL | Top-10 overlap | Loss delta |
|---|---:|---:|---:|
| KV4 + MLP12 Hadamard-plus | 0.152961 | 0.784366 | +0.103611 |
| KV4 + same MLP12 plain Hadamard | 0.167190 | 0.776620 | +0.143428 |

Best previous 16-bus beam frontier swap:

| Stack | KL | Top-10 overlap | Loss delta |
|---|---:|---:|---:|
| KV7 + MLP9 Hadamard-plus | 0.142943 | 0.788592 | -0.033544 |
| KV7 + same MLP9 plain Hadamard | 0.136123 | 0.789859 | -0.011471 |
| Previous quality beam frontier | 0.135650 | 0.789718 | -0.017485 |

Observation: Hadamard-plus is a better local MLP boundary primitive and sometimes improves fixed fused stacks, especially heavier MLP stacks. It is not a clean replacement for plain block-Hadamard inside the best fused frontier. The fused allocator still dominates: the best transform depends on the compressed neighbors, not just the local boundary score.

### Current Answer

The "more Hadamard than Hadamard" idea exists in a useful but constrained form:

```text
Hadamard-plus = pre-stage channels by activation/down_proj structure, then apply cheap block Hadamard.
```

It wins the direct MLP boundary test and gives small end-to-end MLP-stack gains at some budgets. It does not yet move the best fused KV+MLP frontier as a global replacement. The next version should make the allocator choose between plain Hadamard and Hadamard-plus per layer under the fused objective, instead of globally swapping one primitive for the other after local calibration.

## Probe 12: Per-Layer Primitive Choice Under Fused Objective

Scripts:

- `scripts/smol135_fused_choice_allocator.py`
- `scripts/smol135_fused_primitive_swap_probe.py`

Goal: let the fused allocator choose whether each MLP bus should use plain block-Hadamard or Hadamard-plus.

### Mixed-Primitive Beam

Result: `results/smol135_fused_choice_allocator_20260513_143531.json`

This beam started from the old compact base:

```text
KV: 28,24,26,14
MLP: 16,17,15,12 using plain block-Hadamard
```

It considered both `plain` and `plus` choices for future MLP layers. The 16-bus result was:

| Stack | KL | Top-10 overlap | Loss delta |
|---|---:|---:|---:|
| mixed-primitive beam | 0.138691 | 0.788592 | -0.013262 |
| previous quality beam frontier | 0.135650 | 0.789718 | -0.017485 |

Observation: the general mixed-choice beam still pruned into a slightly worse path.

### Targeted Primitive Swap on Previous Best Frontier

Result: `results/smol135_fused_primitive_swap_20260513_144304.json`

Base stack:

```text
KV: 28,24,26,14,13,8,17
MLP: 16,17,15,12,13,19,6,14,5 using plain block-Hadamard
```

Base metrics for the same-layer plain stack:

| KL | Top-10 overlap | Loss delta |
|---:|---:|---:|
| 0.136123 | 0.789859 | -0.011471 |

Accepted primitive swaps:

| Swap | New KL | Top-10 overlap |
|---|---:|---:|
| layer 19 -> Hadamard-plus | 0.135344 | 0.791268 |
| layer 16 -> Hadamard-plus | 0.135098 | 0.792113 |
| layer 17 -> Hadamard-plus | 0.134908 | 0.792394 |

Pairwise search among remaining plain layers found no additional improvement.

Updated 16-bus frontier:

| Stack | KL | Top-10 overlap | Loss delta |
|---|---:|---:|---:|
| previous quality beam frontier | 0.135650 | 0.789718 | -0.017485 |
| targeted primitive-swap frontier | 0.134908 | 0.792394 | -0.008671 |

Observation: this is the clean answer. Hadamard-plus is useful as a selective per-layer primitive, not as a global replacement. The allocator needs a primitive-choice phase: first choose compression sites, then test local primitive swaps under the full fused objective.

## Probe 13: Modal Quality Search and HF Artifact Prep

Scripts:

- `modal_fused_policy_search.py`
- `scripts/smol135_fused_policy_search.py`
- `scripts/export_hf_policy_artifact.py`

Status:

- A Modal smoke run completed successfully and wrote `results/modal_smol135_fused_policy_search_20260513_192003.json`.
- The smoke run evaluated 80 states and stopped at its intended budget.
- The best smoke frontier had 8 compressed buses: KV layers `28,24,26,14` and MLP layers `16,17,15,12`, with KL `0.061339`, top-10 overlap `0.871508`, and estimated saved memory `5.437 MB`.
- A spawned Modal `quality16` run completed successfully and was fetched to `results/modal_smol135_fused_policy_search_20260513_211243.json`.
- The Modal wrapper now persists completed remote results to the `smol135-fused-policy-results` Modal volume and writes a local result if the client remains connected.
- `scripts/export_hf_policy_artifact.py` converts a completed result JSON into a HF-style artifact folder containing `compression_config.json`, `manifest.json`, and `README.md`.
- `scripts/apply_fused_policy.py` can load a packaged policy config and patch a loaded SmolLM2-135M model with the selected runtime KV and MLP activation transforms.

Spawned function call:

```text
fc-01KRHD1RXV9MP95DD0Q4VBSK4F
```

Quality16 summary:

| Total buses | KV layers | MLP choices | KL | Top-10 overlap | Loss delta | Saved MB |
|---:|---|---|---:|---:|---:|---:|
| 8 | `28,24,26,14` | `16:plain,17:plus,15:plain,12:plain` | 0.057946 | 0.856197 | -0.035715 | 5.437 |
| 12 | `28,24,26,14,8,13` | `16:plus,17:plain,15:plus,12:plain,13:plus,5:plain` | 0.091972 | 0.822676 | +0.001325 | 8.156 |
| 16 | `28,24,26,14,15,9,13` | `16:plain,17:plus,15:plain,12:plain,5:plus,14:plain,13:plain,19:plain,18:plus` | 0.132454 | 0.799437 | +0.017879 | 9.843 |

The 16-bus quality16 endpoint improves the previous targeted primitive-swap frontier by KL:

```text
previous: 0.134908
quality16: 0.132454
delta: -0.002453
```

Smoke artifact:

```text
hf_artifacts/smol135-fused-policy-smoke
```

Quality16 artifacts:

```text
hf_artifacts/smol135-fused-policy-quality16-8bus
hf_artifacts/smol135-fused-policy-quality16-14bus
hf_artifacts/smol135-fused-policy-quality16-16bus
```

Next action when `quality16` completes:

```powershell
modal run --write-result results\modal_runs\fused-quality16-latest.json modal_fused_policy_search.py::read_latest_persisted --profile quality16
python scripts/export_hf_policy_artifact.py --result <quality16-result-json> --name smol135-fused-policy-quality16
```
