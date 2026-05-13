# Gemma4 PMRA x OrbitQuant Stack Result

Date: 2026-05-13

Profile: `stack64`

Remote result:

```text
/cache/results/orbitquant_gemma4_pmra_stack/stack64_20260513_215129/result.json
```

Local fetched summary:

```text
results/modal_gemma4_pmra_orbit_stack_stack64_latest.json
```

## Setup

Model: `google/gemma-4-E2B-it`

Static PMRA source: `c2_calib_knapsack_mixed` from the Gemma4 PMRA run.

OrbitQuant policy: depth-scaled SmolLM2 14-bus analog.

- KV buses: layers `33,28,30,16,18,11,15`, 3-bit Hadamard, alpha `0.75`
- MLP buses: layers `19,20,18,14,6,16,15`, 2-bit block/preperm Hadamard, alpha `0.375`
- Eval: Wikitext test, 64 prompts, max length 128
- MLP calibration: Wikitext train, 16 prompts, max length 128

## Result

| Variant | NLL | Delta vs FP16 | Top-10 overlap vs FP16 | Payload bpw |
|---|---:|---:|---:|---:|
| fp16 | 14.349727 | 0.000000 | n/a | 16.000000 |
| q3_k_s | 18.045771 | 3.696043 | 0.051563 | 5.326613 |
| q3_k_s + OrbitQuant | 17.921134 | 3.571407 | 0.046875 | 5.326613 |
| PMRA | 12.984226 | -1.365501 | 0.135938 | 5.326613 |
| PMRA + OrbitQuant | 13.818808 | -0.530919 | 0.134375 | 5.326613 |

## Read

The stack is structurally valid: PMRA and OrbitQuant can be applied together on Gemma4, and the runtime hooks survive the PMRA-patched weight state.

The naive depth-scaled OrbitQuant policy costs PMRA `0.834582` NLL on this 64-prompt check. Even with that toll, `PMRA + OrbitQuant` stays far ahead of uniform `q3_k_s` at the same static payload: `13.818808` vs `18.045771`.

In terms of PMRA's advantage over `q3_k_s`, the naive stack preserves about `83.5%` of the NLL gain:

```text
PMRA gain over q3_k_s:              5.061544 NLL
PMRA + OrbitQuant gain over q3_k_s: 4.226962 NLL
```

The layer policy is the weak point. This policy was mapped from SmolLM2 by relative depth, not selected on Gemma4 or under PMRA weights. The next useful experiment is a PMRA-state OrbitQuant allocator: score individual Gemma4 KV and MLP buses under the PMRA-patched model, then rerun the fused stack with Gemma4-native selected layers.

## Memory

The fetched wrapper reports `50.09 MiB` saved at context length 8192 with 64 MLP live tokens. The audited Gemma4 MLP intermediate dimensions show several selected layers at `12288`, so the corrected selected-policy estimate is about `53.38 MiB` at the same context:

```text
KV saved:  45.50 MiB
MLP saved:  7.88 MiB
Total:     53.38 MiB
```

At longer context, the KV component dominates and scales linearly with context length.
