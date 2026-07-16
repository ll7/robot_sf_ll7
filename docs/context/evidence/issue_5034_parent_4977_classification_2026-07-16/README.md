<!-- AI-GENERATED (robot_sf#5034/#5892, 2026-07-16) - NEEDS-REVIEW -->
# Job 13516 control-action-latency disposition

Plain-language summary: the full fixed-scope campaign is no longer blocked on execution. Job 13516
produced all 7,344 declared episode rows, and the latency subset passed exact cell coverage, but two
planner groups are adapter-backed and the SNQI inputs omit two optional terms. The result therefore
remains a clearly bounded internal diagnostic rather than a paper-facing or pure-native result.

## Claim boundary

**Internal fixed-scope latency-sensitivity result, native execution only, not paper-facing;
fallback/degraded cells are caveats, not successes.**

- Evidence status: `diagnostic-only`.
- Job: `13516` (`5034c-issue5034-latency-sweep`).
- Execution commit: `c153848d7be2851b5c5e89c11055bf96ea778a84`.
- Raw-row SHA-256:
  `6b34e690dfe6cc1ccccd9cd19bde8b3f6a3501bbc1b0a0b44639e151557b4134`.
- Exact scope: 7,344 unique rows; 153 run cells; 48 scenarios; seeds 111, 112, 113;
  planner groups ORCA, default social force, and minimal hybrid rule.
- Exact latency subset: 1,296/1,296 unique expected rows; no missing, extra, or duplicate
  planner/latency/seed/scenario cell.
- Fallback/degraded/unavailable rows: 0/0/0.

## Social Navigation Quality Index result

The Social Navigation Quality Index (SNQI) point-estimate robustness order, from least-negative to
most-negative slope per additional 100 ms-equivalent delay, is:

1. `default_social_force`: -0.002974 SNQI/100 ms, 95% interval [-0.009792, 0.003717].
2. `hybrid_rule_v0_minimal`: -0.005287, 95% interval [-0.011916, 0.001002].
3. `orca`: -0.013762, 95% interval [-0.019933, -0.007928].

Only the default-social-force versus ORCA slope difference clears the paired 95% interval. The
other pairwise ordering statements remain uncertain. ORCA is the only individual planner with a
degradation interval wholly below zero.

## Disposition and caveats

The execution-capacity blocker recorded in the 2026-07-15 classification is resolved. The strict
native-only boundary is not satisfied for the complete three-planner comparison: default social
force is labeled native, while ORCA and the hybrid planner are labeled adapter. Their rows are
retained as explicitly labeled internal diagnostic evidence and are not counted as native success.

The campaign did not emit `force_exceed_events` or `jerk_mean`; canonical SNQI-v0 neutral defaults
apply to those terms. The generated promotion packet also recorded a different Git head from the
job execution context, so the execution-context commit and checksummed plan are the authoritative
run provenance. These limitations prevent promotion beyond the claim boundary above.

Compact evidence:
[`docs/context/evidence/issue_5034_control_action_latency_sweep/`](../issue_5034_control_action_latency_sweep/).
