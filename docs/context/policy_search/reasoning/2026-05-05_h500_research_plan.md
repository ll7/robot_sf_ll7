# H500 Research Plan (2026-05-05)

## Goal

Turn the h500 result into the next bounded policy-search workstream without overstating the
current evidence. The current h500 leader is strong on route completion, but the remaining failures
and near-miss rates show that the next gains need better behavior, not only longer horizons.

## Evidence

- H500 synthesis:
  `docs/context/policy_search/reports/2026-05-05_full_matrix_h500_analysis.md`
- Horizon recommendations:
  `docs/context/policy_search/reports/2026-05-05_h500_horizon_recommendations.md`
- Scenario horizon YAML:
  `configs/policy_search/scenario_horizons_h500.yaml`
- Strict gate reports:
  `docs/context/policy_search/reports/promotions/2026-05-05_full_matrix_h500_strict_gate/`
- Clean rerun handoff:
  `docs/context/policy_search/SLURM/004_h500_leader_clean_rerun.md`

## Workstream A: Deadlock And Route-Local-Minimum Recovery

Target scenarios:

- `classic_merging_medium`
- `classic_station_platform_medium`
- `francis2023_narrow_doorway`
- low-success long-horizon cases such as `classic_merging_low` and cross-trap variants

Candidate direction:

- Keep the current hard static and dynamic safety filters.
- Add deliberate recovery modes only after a sustained low-progress window:
  retreat/recenter, route waypoint reacquisition, and turn-in-place with hysteresis.
- Require per-step diagnostics for recovery activation, selected command source, progress-window
  deltas, and rejection counts.

Proof gate:

- A targeted h500 blocker slice must show at least one newly solved no-success scenario without
  raising collision rate above the strict h500 incumbent envelope.
- A full `full_matrix_h500` rerun is required before any promotion claim.

## Workstream B: Comfort-Preserving High Success

Problem:

- The top h500 candidates reach high success but retain near-miss rates around `0.41-0.42`.

Candidate direction:

- Add speed-dependent near-miss pressure instead of globally slowing the robot.
- Gate comfort pressure by scenario affordance: tight static bottlenecks should not receive the
  same clearance pressure as open group crossing.
- Track `near_miss_rate`, `mean_min_distance`, `avg_speed`, and timeout delta together; do not
  accept lower near-miss exposure if it merely turns successes into timeouts.

Proof gate:

- Compare against `hybrid_rule_v3_fast_progress` and `scenario_adaptive_hybrid_orca_v1` on a
  pinned h500 run.
- Require either lower near-miss rate at no success loss, or equal success with fewer intrusive
  near-miss failures.

## Workstream C: Selector With Safety Accounting

Problem:

- `scenario_adaptive_hybrid_orca_v1` is the h500 aggregate leader, but its advantage over
  `hybrid_rule_v3_fast_progress_static_escape` is only one episode and it misses the strict
  collision gate by one collision.

2026-05-06 update:

- `scenario_adaptive_hybrid_orca_v2_collision_guard` keeps the v1 selector everywhere except
  `classic_merging_low`, where it disables static-escape/recenter extras.
- Targeted micro-sweeps showed tuned ORCA does not repair the two first-step dynamic-deadlock
  collisions (`classic_cross_trap_high` seed `112`, `francis2023_circular_crossing` seed `111`).
- Full h500 evidence: v2 reaches `0.9028` success and `0.0139` collision, passing the strict
  `nominal_sanity` gate while trading away one `classic_merging_low` success from v1.
- Decision: treat v1 as the experimental raw-success leader and v2 as the primary strict-gate
  h500 promotion candidate.

Candidate direction:

- Treat scenario overrides as auditable interventions.
- Require a per-scenario delta table before adding or retaining any override:
  success delta, collision delta, near-miss delta, and mean completion-step delta.
- Prefer overrides that solve a named blocker without changing other scenarios.

Proof gate:

- Strict promotion reports must show whether the selector is a leader-only candidate or a
  strict-gate-clean candidate.
- Overrides that add collisions must be revised or scoped narrower.

## Workstream D: MPC As A Proposer Behind Hard Safety

Problem:

- `mpc_clearance_sampler_v1` has the lowest h500 near-miss exposure in this set, but collision
  rate is too high for promotion.

Candidate direction:

- Reuse MPC clearance or trajectory scoring only as a proposal source.
- Execute proposals through the hybrid rule planner's hard static and dynamic rejection filters.
- Keep diagnostics that distinguish proposal quality from accepted executable command.

Proof gate:

- Target a comfort slice first; do not run a full h500 matrix until static-collision regressions
  are eliminated on blocker scenarios.

## Decision Boundary

Do not add new registered promotion candidates until a smoke or targeted blocker slice has proved
that the candidate runs in the repository. Research notes may describe candidate ideas, but the
registry should remain evidence-backed.
