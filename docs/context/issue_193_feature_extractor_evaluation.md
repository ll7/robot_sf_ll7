# Issue 193: Feature Extractor Evaluation

**Issue**: [#193 – Evaluate current feature extraction in the robot environment](https://github.com/ll7/robot_sf_ll7/issues/193)
**Branch**: `codex/193-feature-extractor-evaluation`
**Date**: 2026-04-16
**Machine**: imech156-u — AMD Ryzen 9 3950X, RTX 3070 8 GB, CUDA 12.8, Ubuntu 24.04

---

## Summary

Four extractors were compared: the original `DynamicsExtractor` against `MLPFeatureExtractor`,
`LightweightCNNExtractor`, and `AttentionFeatureExtractor`. Both a static GPU throughput
microbenchmark and a 32 000-step PPO training run (seed 42) were executed on the same machine.

**Recommendation: replace the default with `mlp_small` for new PPO training runs.**
The original `DynamicsExtractor` remains the only safe choice when loading pre-trained checkpoints
(the module path is hard-coded into SB3 policy files).

---

## Reproducible Commands

**Static microbenchmark** (seconds, no training required):

```bash
# Default: batch=256, 500 reps, auto-detects GPU
uv run python scripts/tools/benchmark_feature_extractors.py

# Save JSON artefact
uv run python scripts/tools/benchmark_feature_extractors.py \
    --batch 256 --reps 500 --out output/bench_fe_issue193.json
```

**Full training comparison**:

```bash
uv run python scripts/multi_extractor_training.py \
    --config configs/scenarios/multi_extractor_eval_193.yaml
```

**Smoke / CI mode** (dramatically shorter):

```bash
ROBOT_SF_MULTI_EXTRACTOR_TEST_MODE=1 \
uv run python scripts/multi_extractor_training.py \
    --config configs/scenarios/multi_extractor_eval_193.yaml
```

---

## Static Microbenchmark (GPU, batch=256)

Measured on RTX 3070, CUDA 12.8, PyTorch with `torch.no_grad()`, 500 reps after 50 warm-up
iterations.

Extractor parameters here are **extractor-only** (not the full policy network).

| Extractor            | Extractor params | Features dim | Throughput (obs/s) | Latency / batch (ms) | Speedup vs baseline |
|----------------------|-----------------|--------------|-------------------:|---------------------:|--------------------:|
| `dynamics_original`  | 5,680           | 89           | 828,225            | 0.309                | 1.00×               |
| `mlp_small`          | 23,176          | 40           | 1,195,111          | 0.214                | **1.44×**           |
| `lightweight_cnn`    | 3,840           | 272          | 610,222            | 0.420                | 0.74×               |
| `attention_small`    | 5,904           | 48           | 397,603            | 0.644                | 0.48×               |

**Interpretation**: `mlp_small` is the fastest extractor despite having more parameters — its
linear layers map more efficiently to GPU GEMM kernels than `dynamics_original`'s Conv1D stack.
`attention_small` is the slowest (2× slower), consistent with the overhead of multi-head attention
over a small 1D sequence. `lightweight_cnn` sits between them but has a high feature dimensionality
(272) relative to its cost.

---

## PPO Training Comparison (32 000 steps, seed=42, RTX 3070)

Config: `configs/scenarios/multi_extractor_eval_193.yaml`
Artifact run: `output/tmp/multi_extractor_training/20260416-075304-eval_193/`

Eval every 4 000 steps, 5 episodes per eval. Total policy parameters include both extractor and
policy head.

| Extractor           | Total policy params | Best eval reward | Final eval reward | Notes                         |
|---------------------|--------------------:|-----------------|------------------|-------------------------------|
| `dynamics_original` | 50,677              | −5.42           | −6.16            | High variance; no clear trend |
| `mlp_small`         | 68,525              | −1.13           | **−1.13**        | Improving trend; ends at best |
| `lightweight_cnn`   | 153,157             | −4.20           | −64.09           | **Diverges at 28 K steps**    |
| `attention_small`   | ~68,000 est.        | −10.19          | incomplete       | Run stopped; ~28 K steps seen |

### Eval reward curves (mean over 5 episodes)

```text
Step    dynamics_original   mlp_small   lightweight_cnn   attention_small
 4000   -31.29              -23.14      -26.22            -65.19
 8000    -6.86              -22.97      -18.97            -21.11
12000   -13.32              -23.49       -4.20            -33.89
16000   -29.23               -8.95      -21.25            -19.18
20000    -5.42              -35.31      -19.71            -16.95
24000   -32.40              -39.12      -22.88            -10.19
28000   -20.46               -2.59      -12.96            -11.60
32000    -6.16               -1.13      -64.09            (no data)
```

**Interpretation**:
- `mlp_small`: clear improvement in the final 8 K steps; ends at its personal best.
- `dynamics_original`: high variance throughout with no consistent improvement direction.
- `lightweight_cnn`: **diverges** sharply at the final eval (−64 from a best of −4.2). This is a
  training instability flag; it should not be used as a default.
- `attention_small`: slow early convergence, incomplete run, highest inference cost. Not suitable
  as a default at this training budget.

---

## Caveats and Limitations

1. **32 000 steps is early-stage PPO**. With the default `n_steps=2048`, this is only ~15 rollout
   batches per extractor. The trends are indicative but not conclusive for final policy quality.
   A full comparison campaign would require ≥500 K steps.
2. **Single seed**. Variance between runs is high (seen in dynamics_original's eval curve). A
   three-seed comparison would reduce noise.
3. **attention_small run terminated before 32 000 steps** due to background job interruption.
   Results at 28 K steps are available; the final eval is missing.
4. **`lightweight_cnn` divergence** at 32 K may be an optimizer interaction (high feature dim
   × policy head scaling). This warrants its own investigation before promotion.

---

## Bug Fixed During Evaluation

`robot_sf/training/hardware_probe.py:44` — `torch.version` is a module, not a dict, so
`.get("cuda", None)` raised `AttributeError`. Fixed:

```python
# before
cuda_version = getattr(torch, "version", {}).get("cuda", None)
# after (safer — avoids AttributeError if torch.version is missing)
version_mod = getattr(torch, "version", None)
cuda_version = getattr(version_mod, "cuda", None) if version_mod is not None else None
```

---

## Recommendation

| Criterion                  | `dynamics_original` | `mlp_small`     |
|----------------------------|---------------------|-----------------|
| Inference throughput       | baseline            | **+44% faster** |
| 32 K PPO best reward       | −5.42               | **−1.13**       |
| Training stability         | variable            | **improving**   |
| Backward compat (checkpts) | **required**        | new runs only   |

**Decision**:
- **For new PPO training runs**: use `mlp_small` as the default feature extractor.
- **For loading existing checkpoints**: keep `DynamicsExtractor` as-is (module path is embedded in
  serialised policy files — changing it breaks loading).
- **Defer** `lightweight_cnn` and `attention_small` promotion to dedicated follow-up issues with
  longer training campaigns and multi-seed comparisons.

---

## Follow-up Issues Needed

1. **Promote `mlp_small` as the new default** in `environment_factory.py` and canonical PPO
   configs — requires updating `policy_kwargs` defaults and verifying backward compat shims for
   checkpoint loading.
2. **Investigate `lightweight_cnn` divergence** at ≥28 K steps (potential learning-rate or
   gradient-norm interaction with large feature dim).
3. **Full multi-seed evaluation campaign** (3 seeds × 500 K steps) to validate `mlp_small`
   superiority beyond the early-training horizon.
4. **`attention_small` re-run** with full 32 K steps to complete the dataset.

---

## Related Files

- `robot_sf/feature_extractor.py` — `DynamicsExtractor` (original; move-protected)
- `robot_sf/feature_extractors/mlp_extractor.py` — `MLPFeatureExtractor`
- `robot_sf/feature_extractors/lightweight_cnn_extractor.py` — `LightweightCNNExtractor`
- `robot_sf/feature_extractors/attention_extractor.py` — `AttentionFeatureExtractor`
- `robot_sf/feature_extractors/config.py` — `FeatureExtractorConfig` / presets
- `robot_sf/training/hardware_probe.py` — bug fixed in this PR
- `configs/scenarios/multi_extractor_eval_193.yaml` — evaluation config (added in this PR)
- `scripts/multi_extractor_training.py` — comparison runner
- `tests/test_feature_extractors.py` — unit + integration tests for all extractors
