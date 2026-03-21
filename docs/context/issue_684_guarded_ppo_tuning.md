# Issue 684 Guarded PPO Tuning Note

## Goal

Test whether `guarded_ppo` can recover meaningful goal-reaching without discarding the safety signal
observed in Issue 602.

## Scope

This issue is deliberately config-only:
- no PPO retraining
- no semantic rewrite of the guard logic
- no new fallback planner family

The only question is whether threshold and fallback-profile tuning can move `guarded_ppo` from
"freezes too often" toward a usable safety/success tradeoff.

## Compared Profiles

- `ppo`
  - canonical baseline from `configs/baselines/ppo_15m_grid_socnav.yaml`
- `guarded_ppo`
  - current benchmark profile from `configs/algos/guarded_ppo_camera_ready.yaml`
- `guarded_ppo_relaxed_v1`
  - `configs/algos/guarded_ppo_relaxed_v1.yaml`
- `guarded_ppo_relaxed_v2`
  - `configs/algos/guarded_ppo_relaxed_v2.yaml`

## Hypothesis

Issue 602 showed a real safety signal but almost total success collapse. The most likely cause was an
overly conservative guard envelope:
- near-field activation too early
- pedestrian clearance thresholds too high
- TTC cutoff too high
- fallback profile not progress-seeking enough once intervention triggers

The relaxed variants reduce those guard thresholds and make the fallback controller more willing to
continue making goal progress.

## Validation Command

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_guarded_ppo_tuning_compare.yaml \
  --label issue684_guarded_ppo_tuning_compare \
  --log-level WARNING
```

## Canonical Artifact

- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_guarded_ppo_tuning_compare_issue684_guarded_ppo_tuning_compare_20260321_190143`

## Result

| Planner | Success | Collisions | SNQI | Runtime (s) | Near Misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ppo` | `0.2695` | `0.1773` | `-0.3512` | `110.0200` | `3.8156` |
| `guarded_ppo` | `0.0071` | `0.0922` | `-0.2197` | `139.2858` | `2.1631` |
| `guarded_ppo_relaxed_v1` | `0.0213` | `0.0993` | `-0.2804` | `124.6702` | `2.7589` |
| `guarded_ppo_relaxed_v2` | `0.0638` | `0.0780` | `-0.3513` | `108.7586` | `3.6383` |

## Interpretation

`guarded_ppo_relaxed_v2` is the best tuned variant in this pass.

What improved relative to the original guarded profile:
- success recovered from `0.0071` to `0.0638`
- runtime dropped from `139.2858s` to `108.7586s`
- collision rate improved from `0.0922` to `0.0780`

What still failed relative to the actual benchmark leaders:
- success remains far below canonical `ppo` at `0.2695`
- collision rate is still worse than canonical `orca` from Issue 602 (`0.0496`)
- SNQI ended effectively tied with PPO (`-0.3513` vs `-0.3512`), so the safety gain was not strong enough to create a better overall tradeoff

The relaxed tuning demonstrates that the original guard was over-conservative and recoverable, but it does not produce a profile that is benchmark-strong enough to promote.

## Verdict

Keep `guarded_ppo_relaxed_v2` only as an internal experimental reference if needed.

Do not promote any guarded-PPO profile to a headline benchmark row. Config-only tuning was enough to show the direction of the tradeoff, but not enough to produce a planner that beats the current best baselines.
