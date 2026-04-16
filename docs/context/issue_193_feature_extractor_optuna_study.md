# Issue #193 — Feature Extractor Optuna Study

**Related issue:** #193  
**Branch:** `codex/193-feature-extractor-evaluation`  
**Status:** Local sweep infrastructure implemented; full SLURM runs pending.

---

## Background and Motivation

The 32 k-step comparison in the initial evaluation (branch `codex/193-feature-extractor-evaluation`,
commit `645a9835`) produced promising directional signals but insufficient evidence to recommend
an architecture change:

- `mlp_small` was 44 % faster on GPU inference and scored better reward (−1.13 vs −5.42),
- `lightweight_cnn` diverged,
- `DynamicsExtractor` (baseline) remained the most stable.

32 k steps is too short to distinguish learning trajectory from initialisation variance.
The study needs at least **10 M steps** for reliable comparisons, and **4 M steps** for
a cost-effective SLURM pre-screen.  An Optuna sweep also enables systematic coverage of
architecture variants (size × type) rather than hand-picked ablations.

### Special interest: temporal feature extractors

The dynamic pedestrian environment is partially observable; an agent that can retain context
across steps should outperform a purely reactive policy.  The primary candidate is LSTM-based
feature extraction.  Note the scope limitation under standard PPO (see below).

---

## Implementation

### New files

| Path | Purpose |
|------|---------|
| `robot_sf/feature_extractors/lstm_extractor.py` | `LSTMFeatureExtractor` — treats the ray array as a 1-D sequence |
| `scripts/training/optuna_feature_extractor.py` | Optuna study runner (local + SLURM mode) |
| `SLURM/submit_feature_extractor_sweep.sh` | Shell launcher that submits N independent SLURM jobs |
| `configs/training/ppo/feature_extractor_sweep_base.yaml` | Base config for the sweep (classic ray/drive-state obs) |

### Changed files

| Path | Change |
|------|--------|
| `robot_sf/feature_extractors/__init__.py` | Export `LSTMFeatureExtractor` |
| `robot_sf/feature_extractors/config.py` | Add `LSTM` enum variant, registry entry, `lstm_small`/`lstm_medium` presets |
| `scripts/training/train_ppo.py` | Extend `_resolve_policy_kwargs` to dispatch all extractor types by name |

---

## LSTM extractor: scope and limitation

`LSTMFeatureExtractor` treats the N-element ray array as a sequence of N scalars (one per
bearing angle).  The LSTM learns sequential spatial patterns within the current observation —
corridor widths, pedestrian clusters, arc shapes.

**It does NOT provide cross-step temporal memory under standard PPO.**  SB3's `PPO` zeros the
hidden state before every forward pass.  True step-to-step retention requires `RecurrentPPO`
from `sb3_contrib`, which is not currently a project dependency.  Adding it is a deferred
follow-up (see risks below).

---

## Search space

| Parameter | Values |
|-----------|--------|
| `extractor_type` | dynamics, mlp, attention, lightweight_cnn, lstm |
| `arch_size` | small, medium, large |
| `policy_arch_size` | small=[64,64], medium=[128,128], large=[256,256] |
| `dropout_rate` | float [0.0, 0.3] |

The `_extractor_kwargs` function in `optuna_feature_extractor.py` maps
`(extractor_type, arch_size, dropout_rate)` → concrete kwargs for each extractor.

---

## Running the sweep

### Local smoke test (32 k steps, ~minutes)

```bash
uv run python scripts/training/optuna_feature_extractor.py \
    --config configs/training/ppo/feature_extractor_sweep_base.yaml \
    --trials 10 \
    --trial-timesteps 32000 \
    --disable-wandb \
    --log-level INFO
```

This runs 10 trials in-process, prints per-extractor FPS, and flags slow candidates.
Use it to verify the pipeline end-to-end before submitting SLURM jobs.

### Full SLURM run (4 M steps, GPU cluster)

```bash
# 1. Create study and submit 20 SLURM jobs
bash SLURM/submit_feature_extractor_sweep.sh \
    --trials 20 \
    --timesteps 4000000 \
    --partition gpu \
    --study-name feat_sweep_4m \
    --time 08:00:00

# 2. Monitor progress
uv run python scripts/tools/inspect_optuna_db.py \
    --storage output/optuna/feat_extractor/feat_sweep_4m.db

# 3. After jobs complete, re-run the script with --trials 0 to print the FPS summary
uv run python scripts/training/optuna_feature_extractor.py \
    --config configs/training/ppo/feature_extractor_sweep_base.yaml \
    --trials 0 \
    --storage sqlite:///output/optuna/feat_extractor/feat_sweep_4m.db \
    --study-name feat_sweep_4m
```

### Excluding slow candidates from the full run

After the local smoke test flags slow types (e.g., `attention large`):

```bash
bash SLURM/submit_feature_extractor_sweep.sh \
    --trials 20 \
    --timesteps 4000000 \
    --exclude attention \
    --study-name feat_sweep_4m_noattn
```

---

## Decision criteria for promoting a candidate

A candidate is eligible for promotion to a new default extractor when:
1. It completes **≥ 10 M steps** training (SLURM full run or dedicated train run).
2. It achieves `success_rate ≥ 0.85` and `collision_rate ≤ 0.08` in a 100-episode evaluation.
3. Its throughput is **≥ 80 % of the DynamicsExtractor FPS** (otherwise the wall-clock cost
   of long training runs becomes prohibitive).
4. It is retested on the canonical benchmark suite to confirm no regression on `snqi` or
   `path_efficiency`.

Do not adopt based on 32 k or 4 M step results alone — those are pre-screening, not proof.

---

## Known risks and deferred follow-up

| Risk | Mitigation |
|------|-----------|
| LSTM has no true temporal memory under PPO | Deferred: add `sb3_contrib.RecurrentPPO` support as a follow-up issue |
| 4 M steps may still be too short for attention/LSTM to converge | Accept — treat 4 M as candidate shortlist; full evidence requires 10 M+ |
| SQLite contention under heavy SLURM parallelism | Use `--storage postgresql://...` for > 20 concurrent jobs |
| `_EXTRACTOR_TYPES` monkey-patched in-process | Harmless for sequential local runs; SLURM workers are independent processes |

---

## Predecessor note

- `docs/context/issue_193_feature_extractor_evaluation.md` (if it exists) — initial 32 k
  ablation results that motivated this study.
