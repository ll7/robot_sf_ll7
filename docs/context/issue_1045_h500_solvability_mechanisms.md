# Issue 1045 H500 Solvability Mechanisms

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1045>
Follow-up planning issue: <https://github.com/ll7/robot_sf_ll7/issues/1044>

## Goal

Analyze when and why the h500 scenario-horizon benchmark makes previously unfinished fixed-horizon
planner-scenario cells solvable, without overstating causal mechanisms that require per-step traces
or videos.

## Evidence

Primary source:

- `docs/context/evidence/issue_1023_candidate_augmented_local_full_2026-05-06/reports/fixed_vs_scenario_horizon_candidates_comparison.json`

Generated issue #1045 evidence:

- `docs/context/evidence/issue_1045_h500_solvability_mechanisms_2026-05-07/h500_solvability_mechanisms.md`
- `docs/context/evidence/issue_1045_h500_solvability_mechanisms_2026-05-07/h500_solvability_mechanisms.json`
- `docs/context/evidence/issue_1045_h500_solvability_mechanisms_2026-05-07/h500_solvability_cases.csv`
- `docs/context/evidence/issue_1045_h500_solvability_mechanisms_2026-05-07/h500_solvability_family_rollup.csv`

Repro command:

```bash
python scripts/tools/analyze_h500_solvability_mechanisms.py \
  docs/context/evidence/issue_1023_candidate_augmented_local_full_2026-05-06/reports/fixed_vs_scenario_horizon_candidates_comparison.json \
  --output-dir docs/context/evidence/issue_1045_h500_solvability_mechanisms_2026-05-07
```

## Findings

The aggregate comparison identifies 123 planner-scenario cells where h500 converts a fixed-horizon
unfinished-heavy case into at least partial success. The mechanism split is:

| Mechanism | Cases | Interpretation |
|---|---:|---|
| `budget_limited_clean_completion` | 38 | Longer horizon fixes a time-budget artifact without aggregate collision or near-miss increase. |
| `late_clean_completion` | 5 | Longer horizon fixes the timeout cleanly, but the successful run uses much of h500. |
| `exposure_enabled_completion` | 40 | Longer horizon enables success but also increases near-miss exposure. |
| `partial_timeout_relief` | 18 | Longer horizon improves success but leaves unresolved failures. |
| `safety_regressed_completion` | 22 | Longer horizon enables some success but with collision-rate increase. |

The cleanest support for h500 realism is the budget-limited group. Representative examples include
ORCA on `classic_bottleneck_low`, `classic_head_on_corridor_low`, and
`classic_urban_crossing_medium`: fixed horizon was unfinished-heavy, h500 reached success, aggregate
collision and near-miss deltas stayed at zero, and candidate normalized time-to-goal stayed below
0.70.

The largest caveat is that near misses and collisions often rise with the longer horizon. That is
not surprising: a run that no longer times out spends more time moving through dynamic interaction
zones, so it can accumulate more near misses. The aggregate evidence therefore supports reporting
h500 as an exposure-rich sensitivity surface, not as a replacement camera-ready benchmark.

## Waiting Claim Boundary

The current tracked full-campaign bundle does not include enough per-step state history to prove
that successes come mostly from waiting until dynamic obstacles passed. The aggregate categories can
distinguish clean budget relief, exposure-enabled completion, partial timeout relief, and safety
regression, but they cannot separate:

- explicit waiting or yielding,
- delayed but continuous progress,
- route-length budget relief,
- recovery after blockage,
- repeated replanning,
- or risk-taking through denser interaction.

Use `scripts/validation/run_policy_search_step_diagnostics.py` or generated videos on a small
representative slice before making the stronger causal claim in issue #1044 or a follow-up paper.

## Current Conclusion

Keep h500 as a sensitivity and follow-up-paper surface. It is scientifically useful because many
fixed-horizon failures are plausibly time-budget artifacts, but the same longer exposure also
creates more near misses and some collision regressions. The next paper-facing step is a trace-backed
representative slice that tests whether the exposure-enabled cases are actually wait-then-go
behaviors or simply longer interaction through dynamic obstacle fields.

Issue #1049 adds that first representative trace slice in
`docs/context/issue_1049_h500_mechanism_pilot.md`. The pilot supports clean budget relief,
exposure/comfort-pressure increase, and safety-regressed long-horizon exposure examples, but it does
not support a broad claim that h500 wins are mostly wait-then-go behavior.
