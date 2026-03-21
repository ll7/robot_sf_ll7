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

- Pending benchmark completion.

## Result

Pending benchmark completion.

## Interpretation

Pending benchmark completion.

## Verdict

Pending benchmark completion.
