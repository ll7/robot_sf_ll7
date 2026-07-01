# Issue 3653 Job 13175 SNQI Scalarization Diagnostic

This bundle applies the existing Social Navigation Quality Index (SNQI) scalarization-sensitivity export to retrieved job `13175` S20/h500 campaign evidence. It is diagnostic analysis only: not benchmark evidence, not a primary-index claim, and not a paper or dissertation claim edit.

## Source

- Source job: `13175`
- Source campaign: `issue1554_s20_h500_l40s_mem180_20260628`
- Submit-host source: `imech192:/home/luttkule/git/robot_sf_ll7.worktrees/slurm-issue-1554-s20-h500-l40s-mem180-20260628/output/benchmarks/camera_ready/issue1554_s20_h500_l40s_mem180_20260628`
- Retrieved/materialized on: `auxme-imech036`
- Local ignored aggregate: `output/issue1554-s20-h500-l40s-mem180/13175/reports/episodes.jsonl`
- Aggregate sha256: `9e11420d8bd4ed8749700cdfd2a8b16f1cc8dbca0766d0a273adfd624e10101a`

The original episode rows omit `planner_key`. The local aggregate restores `planner_key` from each recorded `runs/<planner>__differential_drive/episodes.jsonl` directory so the export covers the intended nine planner lanes instead of merging `scenario_adaptive_hybrid_orca_v1` into fallback algorithm labels.

## Result

- Episodes: `8640`
- Planners: `9`
- SNQI winner: `ppo`
- Constraints-first winner: `hybrid_rule_v3_fast_progress_static_escape`
- Winner disagreement: `true`
- Pairwise reversals: `5`
- Pairwise disagreement rate: `0.1388888888888889`
- Pareto-front points: `2`
- Top mean absolute contribution term: `w_time`

## Files

- `preflight.json`: ready preflight payload.
- `snqi_scalarization_sensitivity.json`: full diagnostic report.
- `snqi_scalarization_sensitivity_planner_rows.csv`: planner-level ranks and Pareto labels.
- `snqi_scalarization_sensitivity_decision_disagreement.csv`: decision-disagreement summary table.
- `snqi_scalarization_sensitivity.md`: report-ready Markdown summary.
- `snqi_scalarization_sensitivity_pareto.svg`: Pareto-front figure.

Raw episode JSONL, full retrieved output, and Slurm logs remain git-ignored.
