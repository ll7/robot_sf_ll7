# Issue #1949 Pedestrian Wait-Duration Perturbation Evidence

This bundle preserves compact, reviewable evidence for the diagnostic-only
`single_pedestrian_wait_duration_offset` pilot.

## Contents

- `summary.json`: compact tracked summary emitted by
  `scripts/validation/run_scenario_perturbation_criticality_pilot.py`.
- `SHA256SUMS`: checksum for the tracked summary.

## Boundary

Raw episode JSONL, materialized scenario matrices, and local coverage output remain ignored under
`output/`. The pilot is local diagnostic evidence only, not benchmark-strength or paper-facing
evidence.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
uv run python scripts/validation/run_scenario_perturbation_criticality_pilot.py \
  configs/scenarios/perturbations/issue_1610_ped_wait_duration_pilot_v1.yaml \
  --materialized-output-dir output/scenario_perturbations/issue1949_ped_wait_duration/materialized \
  --pilot-output-dir output/scenario_perturbations/issue1949_ped_wait_duration/pilot \
  --seed-limit 4 \
  --horizon 80 \
  --dt 0.1 \
  --workers 1 \
  --planner goal \
  --planner orca \
  --planner scenario_adaptive_hybrid_orca_v2_collision_guard \
  --evidence-summary docs/context/evidence/issue_1949_ped_wait_duration_perturbation_2026-06-01/summary.json
```

## Result

The run completed 9/9 eligible paired rows. Mean completed-pair deltas were `0.0` for min distance,
success, collision, and timeout. Two manifest probes failed closed before materialization:
`francis2023_join_group_wait_h3_p050` had no selected `wait_at` entries, and
`classic_head_on_corridor_low_wait_all_p050` had no explicit `single_pedestrians`.
