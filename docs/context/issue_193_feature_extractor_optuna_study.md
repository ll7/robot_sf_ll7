# Issue #193 — Feature Extractor Optuna Study

**Related issue:** #193  
**Branch:** `codex/193-feature-extractor-evaluation`  
**Status:** 4 M-step array sweep finished by 2026-04-19 with 38 completed Optuna trials and 2
stale timeout trials.  Results are a pre-screening snapshot, not promotion evidence or a final
architecture ranking.

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

## 2026-04-17 SLURM Sweep Result

**Study DB:** `output/optuna/feat_extractor/feat_sweep_4m.db`  
**SLURM logs:** `output/slurm/feat_sweep_*.err` / `output/slurm/feat_sweep_*.out`  
**Submitted from handoff:** `docs/context/issue_193_slurm_handoff.md`

The `feat_sweep_4m` study contains 20 trials:

| State | Count | Interpretation |
|-------|------:|----------------|
| `COMPLETE` | 13 | All completed trials used the same `mlp / large / medium policy` config |
| `FAIL` | 7 | All failed trials used `lightweight_cnn / small / large policy` |

Best recorded trial:

| Trial | Metric | Extractor | Arch | Policy arch | Dropout |
|-------|-------:|-----------|------|-------------|--------:|
| `#2` | `30.4144` | `mlp` | `large` | `medium` (`[128, 128]`) | `0.29097` |

Completed `mlp_large` trial statistics:

| Metric | Value |
|--------|------:|
| Eval return mean | `3.0501` |
| Eval return median | `5.9244` |
| Eval return min / max | `-43.4356` / `30.4144` |
| FPS mean | `459.0` |
| FPS min / max | `333.2` / `800.5` |

### Interpretation

This run should be treated as a **distributed-sweep plumbing proof**, not as evidence that
`mlp_large` is the best feature extractor.  The completed trials have only one distinct completed
hyperparameter set, so there is no meaningful cross-extractor ranking.  The wide return spread for
the repeated `mlp_large` config also reinforces that 4 M steps remains a noisy pre-screen.

The 7 failed trials were not the LSTM OOM described in the SLURM handoff.  They failed because the
`lightweight_cnn` path uses CUDA adaptive average pooling whose backward pass is incompatible with
`torch.use_deterministic_algorithms(True)` on this backend:

```text
adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation
```

The deterministic setting comes from `robot_sf/common/seed.py`.  The current resolution is to relax
determinism for `lightweight_cnn` explicitly rather than treating the extractor as invalid.  That
means the extractor remains usable, but the run must be labeled as intentionally nondeterministic
and must not be compared as a bitwise-reproducible baseline.

### Follow-up boundary

Before drawing architecture conclusions, use a rerun with worker-diversified sampler seeds and the
explicit `lightweight_cnn` nondeterminism warning path.  The candidate-promotion criteria below
still apply unchanged.

---

## 2026-04-17 Low-Priority Rerun

The compromised `feat_sweep_4m` study should remain historical.  A fresh rerun was submitted as:

- **Study name:** `feat_sweep_4m_rerun_lowprio_20260417`
- **Storage:** `output/optuna/feat_extractor/feat_sweep_4m_rerun_lowprio_20260417.db`
- **Partition/QoS/account:** `l40s` / `l40s-gpu` / `mitarbeiter`
- **Priority policy:** `--nice=10000`, observed as `Nice=10000` and `Priority=1`
- **Jobs:** `11670` through `11689`
- **Reason for `l40s`:** `pro6000` was drained at submission time

Submission command:

```bash
bash SLURM/submit_feature_extractor_sweep.sh \
    --trials 20 \
    --timesteps 4000000 \
    --partition l40s \
    --account mitarbeiter \
    --qos l40s-gpu \
    --study-name feat_sweep_4m_rerun_lowprio_20260417 \
    --time 08:00:00 \
    --gpus 1 \
    --cpus 8 \
    --mem 32G \
    --nice 10000 \
    --seed 42 \
    --disable-wandb
```

Monitor:

```bash
squeue --me --format='%i %j %T %P %Q %y %b'
uv run python scripts/tools/inspect_optuna_db.py \
    --db output/optuna/feat_extractor/feat_sweep_4m_rerun_lowprio_20260417.db \
    --study-name feat_sweep_4m_rerun_lowprio_20260417 \
    --top-n 10 \
    --show-params
```

### Superseded

This rerun was replaced before meaningful results accumulated.  The next attempt was a fresh split
submission that probed `a30` first and delayed the remaining workers onto `pro6000` for the
weekend:

- **Study name:** `feat_sweep_4m_split_a30_pro6000_20260417`
- **Storage:** `output/optuna/feat_extractor/feat_sweep_4m_split_a30_pro6000_20260417.db`
- **Partition/QoS/account:** `a30` / `a30-gpu` / `mitarbeiter` for the probe job, then
  `pro6000` / `pro6000-gpu` / `mitarbeiter` for the remaining jobs
- **Priority policy:** `--nice=10000` for both partitions
- **Begin policy:** `--begin=2026-04-18T00:00:00` on the `pro6000` batch
- **Jobs:** `11692` on `a30`, `11693` through `11711` on `pro6000`
- **Worker indexing:** `--worker-offset 1` on the `pro6000` batch so the sampler seeds remain
  unique across the split submission

Submission commands:

```bash
bash SLURM/submit_feature_extractor_sweep.sh \
    --trials 1 \
    --timesteps 4000000 \
    --partition a30 \
    --account mitarbeiter \
    --qos a30-gpu \
    --nice 10000 \
    --study-name feat_sweep_4m_split_a30_pro6000_20260417 \
    --seed 42 \
    --disable-wandb

bash SLURM/submit_feature_extractor_sweep.sh \
    --trials 19 \
    --timesteps 4000000 \
    --partition pro6000 \
    --account mitarbeiter \
    --qos pro6000-gpu \
    --nice 10000 \
    --begin 2026-04-18T00:00:00 \
    --worker-offset 1 \
    --study-name feat_sweep_4m_split_a30_pro6000_20260417 \
    --seed 42 \
    --disable-wandb
```

This split study is also historical as of 2026-04-18.  It contains only the completed `a30` probe
trial (`mlp / large / medium policy`, metric `-2.74468`) and did not become the current result
surface.

---

## 2026-04-18 Active Array Sweep Snapshot

> Superseded by the 2026-04-20 final analysis below.  Kept for handoff history.

**Current study DB:** `output/optuna/feat_extractor/feat_sweep_4m_array.db`  
**Study name:** `feat_sweep_4m_array`  
**SLURM arrays:** `11713` on `pro6000`, `11714` on `a30`  
**SLURM logs:** `output/slurm/sweep_4m_array_11713_*.{out,err}` and
`output/slurm/sweep_4m_array_a30_lp_11714_*.{out,err}`

This is the active study to inspect before submitting anything new.  The four local DB files are
not equivalent:

| DB | Status | Notes |
|----|--------|-------|
| `feat_sweep_4m.db` | Historical | First 20-trial sweep; duplicated one `mlp_large` config and failed `lightweight_cnn` under deterministic CUDA pooling. |
| `feat_sweep_4m_rerun_lowprio_20260417.db` | Superseded | `l40s` low-priority rerun; job `11670` was cancelled after ~9 minutes, leaving no completed trials. |
| `feat_sweep_4m_split_a30_pro6000_20260417.db` | Superseded/stalled | Split rerun surface; only one completed probe trial is present. |
| `feat_sweep_4m_array.db` | Active | Current diversified array sweep; use this for progress and next decisions. |

Observed progress at 2026-04-18 20:58 Europe/Berlin:

| State | Count | Notes |
|-------|------:|-------|
| `COMPLETE` | 28 | Completed across `dynamics`, `mlp`, `attention`, `lightweight_cnn`, and `lstm`. |
| `RUNNING` | 4 | Trials `28`-`31`, all `lstm / large / medium policy`. |

Current top trial:

| Trial | Metric | Extractor | Arch | Policy arch | Dropout |
|-------|-------:|-----------|------|-------------|--------:|
| `#19` | `51.1163` | `dynamics` | `large` | `medium` (`[128, 128]`) | `0.00013` |

Completed-trial aggregate snapshot:

| Extractor | Completed | Mean return | Best return | Mean FPS | Min FPS |
|-----------|----------:|------------:|------------:|---------:|--------:|
| `dynamics` | 6 | `30.432` | `51.116` | `465.7` | `374.7` |
| `lightweight_cnn` | 13 | `34.577` | `48.804` | `498.3` | `377.2` |
| `mlp` | 5 | `20.764` | `34.137` | `436.4` | `351.2` |
| `lstm` | 2 | `26.069` | `30.247` | `301.4` | `142.1` |
| `attention` | 2 | `-5.797` | `-5.797` | `277.3` | `266.3` |

### Current interpretation

Do not submit a new array while `11713`/`11714` are still running or pending.  The active DB already
has broad enough 4 M-step coverage to wait for completion and then shortlist candidates.  The
current snapshot suggests `dynamics` and `lightweight_cnn` deserve follow-up scrutiny, while
`attention` is weak at this budget and `lstm` is still under-sampled because the remaining running
trials are LSTM-heavy.

This is still **not** enough to promote a default extractor.  Treat the active array as a
pre-screen: after it finishes, choose a small shortlist and run longer, cleaner validation rather
than expanding the search immediately.

Recommended next step after array completion:

1. Inspect the final DB with:

   ```bash
   uv run python scripts/tools/inspect_optuna_db.py \
       --db output/optuna/feat_extractor/feat_sweep_4m_array.db \
       --study-name feat_sweep_4m_array \
       --top-n 20 \
       --show-params
   ```

2. Pick two or three candidate configs from the final top set plus the baseline.
3. Run dedicated longer training, preferably 10 M+ steps with multiple seeds, before any default
   extractor recommendation.

---

## 2026-04-20 Final Array Sweep Analysis

**Current source of truth:** `output/optuna/feat_extractor/feat_sweep_4m_array.db`  
**Study name:** `feat_sweep_4m_array`  
**Last DB modification:** 2026-04-19 13:29 Europe/Berlin  
**Queue state checked:** `squeue --me` returned no active jobs on 2026-04-20.

The active array sweep is done from SLURM's perspective.  Optuna still reports two `RUNNING` trials,
but both are stale timeout records rather than active work:

| Trial | Config | Why excluded |
|-------|--------|--------------|
| `#28` | `lstm / large / medium policy`, dropout `0.0701` | Corresponding `a30` task `11714_12` timed out at the 8-hour SLURM limit. |
| `#29` | `lstm / large / medium policy`, dropout `0.0881` | Corresponding `a30` task `11714_13` timed out at the 8-hour SLURM limit. |

Their partial logs contain intermediate metrics, but because Optuna did not commit final values,
they should not be mixed into the ranking.

### Final study status

| State | Count | Interpretation |
|-------|------:|----------------|
| `COMPLETE` | 38 | Valid Optuna evidence for pre-screening. |
| `RUNNING` | 2 | Stale timeout records from `a30`; exclude from ranking. |

The older DBs remain historical:

| DB | Trials | Completed | Best | Use |
|----|------:|----------:|-----:|-----|
| `feat_sweep_4m.db` | 20 | 13 | `30.4144` | Plumbing proof only; duplicated `mlp_large` suggestions and failed CNN deterministic CUDA pooling. |
| `feat_sweep_4m_rerun_lowprio_20260417.db` | 1 | 0 | n/a | Superseded cancelled `l40s` attempt. |
| `feat_sweep_4m_split_a30_pro6000_20260417.db` | 1 | 1 | `-2.7447` | Superseded one-probe split attempt. |
| `feat_sweep_4m_array.db` | 40 | 38 | `51.1163` | Current pre-screening evidence. |

### Top completed trials

| Rank | Trial | Metric | Extractor | Arch | Policy arch | Dropout | FPS |
|-----:|------:|-------:|-----------|------|-------------|--------:|----:|
| 1 | `#19` | `51.116` | `dynamics` | `large` | `[128, 128]` | `0.0001` | `387.7` |
| 2 | `#23` | `48.804` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0527` | `517.8` |
| 3 | `#15` | `48.499` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0099` | `520.1` |
| 4 | `#25` | `48.002` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0537` | `377.2` |
| 5 | `#17` | `44.175` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0008` | `518.9` |
| 6 | `#13` | `43.431` | `dynamics` | `large` | `[256, 256]` | `0.0005` | `374.7` |
| 7 | `#36` | `41.593` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0390` | `358.8` |
| 8 | `#14` | `40.010` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0171` | `524.0` |
| 9 | `#37` | `39.555` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0379` | `369.7` |
| 10 | `#31` | `37.869` | `lstm` | `large` | `[128, 128]` | `0.0341` | `210.7` |

Full active-study trial table:

| Trial | State | Extractor | Arch | Policy arch | Dropout | Return | FPS |
|------:|-------|-----------|------|-------------|--------:|-------:|----:|
| `#19` | `COMPLETE` | `dynamics` | `large` | `[128, 128]` | `0.0001` | `51.116` | `387.7` |
| `#23` | `COMPLETE` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0527` | `48.804` | `517.8` |
| `#15` | `COMPLETE` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0099` | `48.499` | `520.1` |
| `#25` | `COMPLETE` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0537` | `48.002` | `377.2` |
| `#17` | `COMPLETE` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0008` | `44.175` | `518.9` |
| `#13` | `COMPLETE` | `dynamics` | `large` | `[256, 256]` | `0.0005` | `43.431` | `374.7` |
| `#36` | `COMPLETE` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0390` | `41.593` | `358.8` |
| `#14` | `COMPLETE` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0171` | `40.010` | `524.0` |
| `#37` | `COMPLETE` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0379` | `39.555` | `369.7` |
| `#31` | `COMPLETE` | `lstm` | `large` | `[128, 128]` | `0.0341` | `37.869` | `210.7` |
| `#30` | `COMPLETE` | `lstm` | `large` | `[128, 128]` | `0.0755` | `36.730` | `213.6` |
| `#6` | `COMPLETE` | `mlp` | `large` | `[256, 256]` | `0.0390` | `34.137` | `351.2` |
| `#24` | `COMPLETE` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0567` | `33.350` | `533.7` |
| `#7` | `COMPLETE` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0668` | `33.153` | `552.1` |
| `#9` | `COMPLETE` | `lightweight_cnn` | `medium` | `[256, 256]` | `0.1081` | `32.204` | `387.8` |
| `#26` | `COMPLETE` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0556` | `31.630` | `543.0` |
| `#3` | `COMPLETE` | `lstm` | `large` | `[256, 256]` | `0.1774` | `30.247` | `142.1` |
| `#18` | `COMPLETE` | `dynamics` | `small` | `[128, 128]` | `0.0173` | `28.062` | `378.8` |
| `#16` | `COMPLETE` | `lightweight_cnn` | `small` | `[256, 256]` | `0.0222` | `27.848` | `521.7` |
| `#39` | `COMPLETE` | `dynamics` | `small` | `[128, 128]` | `0.0552` | `27.376` | `400.6` |
| `#11` | `COMPLETE` | `dynamics` | `medium` | `[256, 256]` | `0.2310` | `25.896` | `547.3` |
| `#32` | `COMPLETE` | `attention` | `large` | `[64, 64]` | `0.0930` | `24.605` | `311.6` |
| `#10` | `COMPLETE` | `mlp` | `medium` | `[128, 128]` | `0.2264` | `23.623` | `522.8` |
| `#38` | `COMPLETE` | `dynamics` | `small` | `[128, 128]` | `0.0453` | `23.215` | `388.0` |
| `#22` | `COMPLETE` | `lightweight_cnn` | `small` | `[128, 128]` | `0.1227` | `23.005` | `389.2` |
| `#34` | `COMPLETE` | `attention` | `large` | `[64, 64]` | `0.0888` | `22.831` | `149.1` |
| `#33` | `COMPLETE` | `attention` | `large` | `[64, 64]` | `0.0870` | `22.808` | `311.7` |
| `#35` | `COMPLETE` | `attention` | `large` | `[64, 64]` | `0.0963` | `22.736` | `150.0` |
| `#0` | `COMPLETE` | `mlp` | `medium` | `[64, 64]` | `0.0942` | `22.376` | `425.1` |
| `#4` | `COMPLETE` | `lstm` | `small` | `[64, 64]` | `0.1830` | `21.890` | `460.6` |
| `#20` | `COMPLETE` | `dynamics` | `large` | `[128, 128]` | `0.1282` | `21.201` | `560.5` |
| `#27` | `COMPLETE` | `lightweight_cnn` | `small` | `[128, 128]` | `0.0737` | `21.015` | `538.8` |
| `#2` | `COMPLETE` | `mlp` | `large` | `[64, 64]` | `0.2532` | `20.182` | `396.4` |
| `#21` | `COMPLETE` | `lightweight_cnn` | `small` | `[128, 128]` | `0.1275` | `17.802` | `553.1` |
| `#8` | `COMPLETE` | `dynamics` | `medium` | `[128, 128]` | `0.2363` | `12.887` | `545.2` |
| `#5` | `COMPLETE` | `mlp` | `large` | `[128, 128]` | `0.2910` | `3.501` | `486.3` |
| `#1` | `COMPLETE` | `attention` | `medium` | `[256, 256]` | `0.1694` | `-5.797` | `288.2` |
| `#12` | `COMPLETE` | `attention` | `small` | `[64, 64]` | `0.2191` | `-5.797` | `266.3` |
| `#28` | `RUNNING` | `lstm` | `large` | `[128, 128]` | `0.0701` | n/a | n/a |
| `#29` | `RUNNING` | `lstm` | `large` | `[128, 128]` | `0.0881` | n/a | n/a |

### Aggregate by extractor

| Extractor | Completed | Mean return | Min return | Best return | Mean FPS | Min FPS | Max FPS |
|-----------|----------:|------------:|-----------:|------------:|---------:|--------:|--------:|
| `dynamics` | 8 | `29.148` | `12.887` | `51.116` | `447.9` | `374.7` | `560.5` |
| `lightweight_cnn` | 15 | `35.376` | `17.802` | `48.804` | `480.4` | `358.8` | `553.1` |
| `lstm` | 4 | `31.684` | `21.890` | `37.869` | `256.8` | `142.1` | `460.6` |
| `mlp` | 5 | `20.764` | `3.501` | `34.137` | `436.4` | `351.2` | `522.8` |
| `attention` | 6 | `13.564` | `-5.797` | `24.605` | `246.2` | `149.1` | `311.7` |

### Aggregate by repeated config

| Config | Completed | Mean return | Best return | Mean FPS | Interpretation |
|--------|----------:|------------:|------------:|---------:|----------------|
| `lightweight_cnn / small / [128,128]` | 13 | `36.200` | `48.804` | `484.3` | Strongest repeated candidate; high throughput and consistently competitive returns. |
| `dynamics / large / [128,128]` | 2 | `36.159` | `51.116` | `474.1` | Best single trial, but the second replicate dropped to `21.201`, so variance is substantial. |
| `lstm / large / [128,128]` | 2 | `37.300` | `37.869` | `212.2` | Competitive return, but slow and two same-config `a30` attempts timed out. |
| `attention / large / [64,64]` | 4 | `23.245` | `24.605` | `230.6` | Recovered from earlier poor trials but still behind top candidates and relatively slow. |

### Reasoning and recommendation

The sweep should be treated as a candidate filter, not a decision to change defaults.  It used
4 M-step Optuna trials, uneven sampler coverage, and a noisy single-trial objective.  The best
single value is `dynamics / large / [128,128]`, but the best repeated family is
`lightweight_cnn / small / [128,128]`: many of its repeats land in the top half, its mean return is
highest among extractor families, and its throughput is better than the baseline-family mean in
this run.

The initial 32 k result that favored `mlp_small` does not survive this longer pre-screen.  MLP is
still fast, but none of its 4 M trials matched the top `dynamics` or `lightweight_cnn` candidates.
Attention should not be a promotion candidate from these results.  LSTM remains scientifically
interesting, but under standard PPO it does not provide true cross-step memory, runs slower, and
large LSTM variants are close to the wall-time limit on `a30`.

Recommended shortlist for longer validation:

1. `lightweight_cnn / small / [128,128]`, with low dropout around `0.01`-`0.06`.
2. `dynamics / large / [128,128]`, dropout near zero, as the best single-trial baseline-family
   candidate.
3. Current/default `DynamicsExtractor` configuration as a compatibility and regression baseline.
4. Optional: `lstm / large / [128,128]` only if the goal includes temporal/spatial-sequence
   research and the run is placed on hardware/time limits that can finish it reliably.

### Proposed next steps

1. Do not submit another broad Optuna array yet.
2. Run dedicated candidate validation at **10 M+ steps**, preferably 3 seeds per candidate:
   `lightweight_cnn_small_medium-policy`, `dynamics_large_medium-policy`, and the current default.
3. Use a wall time above 8 hours or avoid `a30` for LSTM/large candidates; the stale Optuna records
   show the 8-hour `a30` limit is too tight.
4. Evaluate each completed long-run policy with the promotion gate from this note:
   `success_rate >= 0.85`, `collision_rate <= 0.08`, at least 80 % of baseline FPS, and no
   benchmark regression on `snqi` or `path_efficiency`.
5. Keep `lightweight_cnn` explicitly labeled as nondeterministic on CUDA adaptive-pooling backward
   unless/until that implementation path is changed.

### Validation commands used for this analysis

```bash
squeue --me --format='%i %j %T %P %Q %y %b %S'

uv run python scripts/tools/inspect_optuna_db.py \
    --db output/optuna/feat_extractor/feat_sweep_4m_array.db \
    --study-name feat_sweep_4m_array \
    --top-n 40 \
    --show-params

sqlite3 -header -column output/optuna/feat_extractor/feat_sweep_4m_array.db \
    "select state, count(*) as n from trials group by state order by state;"

sacct -j 11713,11714 \
    --format=JobID,JobName%28,State,ExitCode,Partition,Elapsed,Start,End -P
```

---

## 2026-04-20 12M Fixed-Candidate Submission

The requested 16-hour `pro6000` plan was checked against live SLURM state before submission.
`sinfo -p pro6000` reported a `13:00:00` partition time limit, while `pro6000-gpu` QoS had no
stricter `MaxWall`.  The `pro6000` node was `IDLE+DRAIN` during the daytime with reason
`Node is disabled during daytime`, so the jobs were submitted with the maximum valid `13:00:00`
limit and low priority rather than the invalid `16:00:00` limit.

The 12M hardening batch is a fixed candidate matrix, not a free Optuna sampler:

- Candidate file: `configs/training/ppo/feature_extractor_candidates_12m_issue193.yaml`
- Runner: `scripts/training/fixed_feature_extractor_candidates.py`
- Launcher: `SLURM/submit_feature_extractor_fixed_candidates.sh`
- Study: `feat_extractor_12m_hardening_20260420`
- Storage: `output/optuna/feat_extractor/feat_extractor_12m_hardening_20260420.db`
- SLURM array: `11874`, tasks `0-9%2`
- Partition/QoS/account: `pro6000` / `pro6000-gpu` / `mitarbeiter`
- Priority: `--nice=10000`, observed priority `1`
- Time limit: `13:00:00`

Submission command:

```bash
bash SLURM/submit_feature_extractor_fixed_candidates.sh \
    --candidate-file configs/training/ppo/feature_extractor_candidates_12m_issue193.yaml \
    --study-name feat_extractor_12m_hardening_20260420 \
    --storage sqlite:///output/optuna/feat_extractor/feat_extractor_12m_hardening_20260420.db \
    --partition pro6000 \
    --account mitarbeiter \
    --qos pro6000-gpu \
    --nice 10000 \
    --time 13:00:00 \
    --array-concurrency 2 \
    --disable-wandb
```

Initial queue state after submission:

```text
11874_[0-9%2] feat_12m_fixed PENDING pro6000 priority=1 nice=10000 time=13:00:00
Reason: Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions
```

This pending reason is expected during the daytime drain window.  If it remains unchanged during
the next night window, re-check the node reason with:

```bash
scontrol show node auxme-imech142 | rg 'State=|Reason=|Partitions=|Gres='
```

Candidate allocation:

| Tasks | Candidate | Seeds | Reason |
|-------|-----------|-------|--------|
| `0-3` | `lightweight_cnn / small / [128,128] / dropout=0.05` | `123`, `231`, `1337`, `2026` | Strongest repeated 4M family; extra fourth seed hardens the likely promotion candidate. |
| `4-6` | `dynamics / large / [128,128] / dropout=0.0` | `123`, `231`, `1337` | Best single 4M trial, needs variance check. |
| `7-9` | original/default `DynamicsExtractor` shape (`dynamics`, empty kwargs, `[64,64]`) | `123`, `231`, `1337` | Compatibility and regression baseline. |

LSTM was intentionally left out of this 10-job batch.  The stale 4M Optuna records showed
`lstm / large / [128,128]` timing out under the 8-hour `a30` limit, and extrapolating its 4M
runtime makes a 12M `13:00:00` pro6000 job risky.  If LSTM remains interesting, submit it as a
separate long-wall-time job on a partition that can actually provide the wall time.

Monitor:

```bash
squeue -j 11874 --format='%i %j %T %P %Q %y %b %S %R'

uv run python scripts/tools/inspect_optuna_db.py \
    --db output/optuna/feat_extractor/feat_extractor_12m_hardening_20260420.db \
    --study-name feat_extractor_12m_hardening_20260420 \
    --top-n 10
```

### 2026-04-21 retry for cancelled tasks

Progress check on 2026-04-21 showed:

| Original task | Candidate | State | Action |
|---------------|-----------|-------|--------|
| `11874_6` | `dyn_large_med_s1337` | `CANCELLED by 1002` after `03:02:08` | Original Optuna trial marked `FAIL` with `failure_type=slurm_cancelled`; candidate resubmitted. |
| `11874_7` | `dyn_default_s123` | `CANCELLED by 1002` after `02:57:28` | Original Optuna trial marked `FAIL` with `failure_type=slurm_cancelled`; candidate resubmitted. |
| `11874_8-9` | default dynamics seeds `231`, `1337` | still `PENDING` on `pro6000` | Left in place for the next `pro6000` availability window. |

The retry was moved to `l40s` because `pro6000` daytime drain cancelled the first attempts and the
remaining `pro6000` tasks were still pending for node availability.  `l40s` reported a 3-day
partition limit and no stricter `l40s-gpu` QoS wall-time cap, so the retries use a conservative
24-hour low-priority wall time.

Retry submission:

```bash
sbatch \
    --job-name=feat_12m_retry \
    --array=6-7%2 \
    --time=24:00:00 \
    --cpus-per-task=8 \
    --mem=32G \
    --gres=gpu:1 \
    --output=output/slurm/feat_12m_retry_%A_%a.out \
    --error=output/slurm/feat_12m_retry_%A_%a.err \
    --partition=l40s \
    --account=mitarbeiter \
    --qos=l40s-gpu \
    --nice=10000 \
    --wrap 'uv run python scripts/training/fixed_feature_extractor_candidates.py --candidate-file configs/training/ppo/feature_extractor_candidates_12m_issue193.yaml --candidate-index $SLURM_ARRAY_TASK_ID --study-name feat_extractor_12m_hardening_20260420 --storage sqlite:///output/optuna/feat_extractor/feat_extractor_12m_hardening_20260420.db --log-level WARNING --disable-wandb'
```

Retry job:

| Job | Tasks | Partition | State at submission check |
|-----|-------|-----------|---------------------------|
| `11907` | `6-7%2` | `l40s` | `PENDING`, reason `Priority`; task `6` had estimated start `2026-04-23T09:36:17`. |

Monitor both arrays:

```bash
squeue -j 11874,11907 --format='%i %j %T %P %Q %y %b %M %l %S %R'
```

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
    --db output/optuna/feat_extractor/feat_sweep_4m.db

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
| Concurrent one-trial workers can duplicate early TPE suggestions when launched with identical sampler seeds | Worker commands now pass `--worker-index`; rerun evidence is still needed |
| SQLite contention under heavy SLURM parallelism | Use `--storage postgresql://...` for > 20 concurrent jobs |
| `lightweight_cnn` hits a non-deterministic CUDA adaptive-pooling backward kernel under the global deterministic setting | Relax determinism explicitly for this extractor and emit a loud non-reproducibility warning |
| `_EXTRACTOR_TYPES` monkey-patched in-process | Harmless for sequential local runs; SLURM workers are independent processes |

---

## Predecessor note

- `docs/context/issue_193_feature_extractor_evaluation.md` (if it exists) — initial 32 k
  ablation results that motivated this study.
