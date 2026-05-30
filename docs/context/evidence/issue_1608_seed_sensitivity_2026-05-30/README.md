# Issue #1608 Scenario Seed-Sensitivity Evidence

Date: 2026-05-30

This bundle preserves compact, reviewable derived outputs for issue #1608. The analysis reads the
durable issue #1454 S10/h500 candidate campaign artifacts and does not rerun a benchmark.

## Source Command

```bash
uv run python scripts/tools/analyze_scenario_seed_sensitivity.py \
  --campaign-root docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23 \
  --output-dir docs/context/evidence/issue_1608_seed_sensitivity_2026-05-30
```

## Source Evidence

- Input bundle:
  `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/`
- Source command for the input bundle:
  `configs/benchmarks/issue_1454_s10_scenario_horizons_h500_candidates.yaml`
- Input bundle commit: `4941ac48`
- Raw campaign archive pointer:
  <https://github.com/ll7/robot_sf_ll7/releases/tag/artifact/issue1454-s10-h500-candidates-2026-05-23>

## Result

- Selected planners: top four benchmark-success rows by success, then collision,
  `time_to_goal_norm`, near misses, and planner key:
  - `hybrid_rule_v3_fast_progress_static_escape_continuous`
  - `scenario_adaptive_hybrid_orca_v1`
  - `scenario_adaptive_hybrid_orca_v2_collision_guard`
  - `hybrid_rule_v3_fast_progress_static_escape`
- Scenario rows classified: `48`
- Seed-sensitive scenarios: `25`
- Not seed-sensitive scenarios: `23`
- Inconclusive scenarios: `0`
- Hardest seed id by mean top-planner success: `116`

## Included Files

- `seed_sensitivity_analysis.json`: full structured analysis payload.
- `seed_sensitivity_analysis.md`: compact human-readable report.
- `scenario_seed_sensitivity.csv`: one row per scenario with classification and hard/easy seed ids.
- `seed_difficulty_summary.csv`: one row per scenario/seed with mean top-planner success and
  collision rates.

## Claim Boundary

This is derived diagnostic evidence for scenario prioritization. It is not causal mechanism proof
and is not paper-facing significance evidence. The source campaign is exploratory candidate evidence
with `paper_facing=false`, and SNQI is not used for this issue's planner selection or classification.
