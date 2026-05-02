# Issue #850 PPO Collision Failures

**Related issue:** [#850](https://github.com/ll7/robot_sf_ll7/issues/850)  
**Predecessor:** [Issue #193 Feature Extractor Optuna Study](issue_193_feature_extractor_optuna_study.md)

## Goal

Diagnose why the issue-193 selected `dyn_large_med` PPO candidate failed the promotion gate and
define one narrow mitigation that can be evaluated without reopening a broad feature-extractor
sweep.

## Evidence

Issue #193 hold-out policy analysis selected `dyn_large_med` as the best architecture family but
rejected promotion:

| Candidate | Success | Collision | Ped collision | Obstacle collision |
|-----------|--------:|----------:|--------------:|-------------------:|
| `dyn_large_med_s231` | `0.727` | `0.273` | `0.091` | `0.182` |
| `dyn_large_med_s1337` | `0.727` | `0.273` | `0.091` | `0.182` |
| `dyn_large_med_s123` | `0.667` | `0.333` | `0.076` | `0.258` |

The observed blocker is policy safety rather than feature-extractor capacity.  The repeated
collision hotspots were obstacle/wall collisions in `classic_bottleneck_high`,
`classic_merging_low`, and `classic_merging_medium`, with a smaller pedestrian-collision tail in
doorway, cross-trap, and double-bottleneck cases.

## Added Tooling

`scripts/tools/analyze_policy_collision_failures.py` consumes one or more `policy_analysis_run.py`
output directories and produces a per-run and per-scenario collision split.  It is intended to be
run on the issue-193 artifacts and on any issue-850 mitigation outputs:

```bash
uv run python scripts/tools/analyze_policy_collision_failures.py \
  output/benchmarks/issue193_policy_analysis_dyn_large_med_s231 \
  output/benchmarks/issue193_policy_analysis_dyn_large_med_s1337 \
  output/benchmarks/issue193_policy_analysis_dyn_large_med_s123 \
  --output-md output/analysis/issue850_collision_failures.md \
  --output-json output/analysis/issue850_collision_failures.json
```

## Mitigation Hypothesis

Use a config-first, architecture-fixed mitigation:

- Keep `dyn_large_med`: `feature_extractor=dynamics`, filters `[128, 32, 32, 16]`,
  `policy_net_arch=[128,128]`, dropout `0.0`.
- Switch from the issue-193 `route_completion_v2` reward to stricter `route_completion_v3`.
- Increase collision, near-miss, TTC-risk, timeout, and stagnation penalties.
- Upweight the observed obstacle-collision hotspot scenarios during training.

Config:

```bash
configs/training/ppo/feature_extractor_sweep_dyn_large_med_safety_v3.yaml
```

Canonical hold-out command after training:

```bash
SDL_VIDEODRIVER=dummy MPLBACKEND=Agg uv run python scripts/tools/policy_analysis_run.py \
  --training-config configs/training/ppo/feature_extractor_sweep_dyn_large_med_safety_v3.yaml \
  --policy ppo \
  --model-path <checkpoint.zip> \
  --seed-set eval \
  --max-seeds 3 \
  --output output/benchmarks/issue850_policy_analysis_dyn_large_med_safety_v3 \
  --video-output output/recordings/issue850_policy_analysis_dyn_large_med_safety_v3 \
  --all
```

## Current Limitation

The local workspace does not contain the issue-193 `dyn_large_med` checkpoints or the
`output/benchmarks/issue193_policy_analysis_*` artifacts referenced by the predecessor note, so
the hold-out gate was not rerun in this pass.  Do not present this mitigation as successful until
the command above produces `episodes.jsonl`, `summary.json`, and `report.json` and the promotion
gate is checked against `success_rate >= 0.85` and `collision_rate <= 0.08`.
