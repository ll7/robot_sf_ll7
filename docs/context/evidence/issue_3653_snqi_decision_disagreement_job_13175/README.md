# Issue #3653 SNQI Decision-Disagreement Diagnostic

This bundle applies the existing Social Navigation Quality Index (SNQI) scalarization-sensitivity, Pareto-front, and decision-disagreement export to the hydrated completed job `13175` S20/h500 diagnostic campaign surface. It is diagnostic-only evidence on a checksum-pinned surface; it does not promote a benchmark, safety-ranking, scenario-horizon, dissertation, or paper claim.

## Source

- Source job: `13175`
- Source campaign: `issue1554_s20_h500_l40s_mem180_20260628`
- Submit-host source: `imech192:/home/luttkule/git/robot_sf_ll7.worktrees/slurm-issue-1554-s20-h500-l40s-mem180-20260628/output/benchmarks/camera_ready/issue1554_s20_h500_l40s_mem180_20260628`
- Hydrated aggregate: `output/issue1554-s20-h500-l40s-mem180/13175/reports/episodes.jsonl`
- Aggregate SHA-256: `fd15480d6892dd634e374fb9f79e1e3600d24c88604d9ff05f33d8227b4e6460`
- Episode rows: `8640`

The original per-run episode rows omit `planner_key`. The hydrated aggregate restores `planner_key` from each recorded `runs/<planner>__differential_drive/episodes.jsonl` directory so the export covers the intended nine planner lanes.

## Result

- Planners: `9`
- Episodes: `8640`
- SNQI winner: `ppo`
- Constraints-first winner: `hybrid_rule_v3_fast_progress_static_escape`
- Winner disagreement: `true`
- Pairwise reversals: `5`
- Pairwise disagreement rate: `0.1388888888888889`
- Pareto-front points: `2`

## Claim Boundary

The correct claim is narrow: on job `13175`, the SNQI scalarization diagnostic has been applied to a checksum-pinned S20/h500 campaign surface and produced populated Pareto-front and decision-disagreement artifacts. The bundle does not validate SNQI as a primary index and does not promote paper/dissertation benchmark claims.

## Files

- `preflight.json`: SNQI scalarization-sensitivity readiness preflight.
- `snqi_scalarization_sensitivity.json`: canonical export report.
- `snqi_scalarization_sensitivity_planner_rows.csv`: planner-level sensitivity rows.
- `snqi_scalarization_sensitivity_decision_disagreement.csv`: decision-disagreement table.
- `snqi_scalarization_sensitivity.md`: Markdown summary.
- `snqi_scalarization_sensitivity_pareto.svg`: Pareto-front plot.
