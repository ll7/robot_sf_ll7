# Issue 1045 H500 Solvability Mechanism Evidence

Compact aggregate evidence for issue #1045, generated on 2026-05-07 from the tracked issue #1023
candidate-augmented fixed-vs-scenario-horizon comparison.

Source comparison:

```bash
docs/context/evidence/issue_1023_candidate_augmented_local_full_2026-05-06/reports/fixed_vs_scenario_horizon_candidates_comparison.json
```

Command:

```bash
python scripts/tools/analyze_h500_solvability_mechanisms.py \
  docs/context/evidence/issue_1023_candidate_augmented_local_full_2026-05-06/reports/fixed_vs_scenario_horizon_candidates_comparison.json \
  --output-dir docs/context/evidence/issue_1045_h500_solvability_mechanisms_2026-05-07
```

Files:

- `h500_solvability_mechanisms.md` - human-readable summary and interpretation boundary.
- `h500_solvability_mechanisms.json` - full generated analysis with thresholds and all cases.
- `h500_solvability_cases.csv` - timeout-to-success planner-scenario cells.
- `h500_solvability_family_rollup.csv` - scenario-family aggregate rollup.

Evidence boundary:

- This bundle is aggregate comparison evidence. It can show where h500 converts fixed-horizon
  unfinished runs into successes, and whether that conversion coincides with collision or near-miss
  increases.
- It cannot prove that a planner waited until dynamic obstacles passed, because the source bundle
  does not include per-step traces or videos for the compared full campaigns.
