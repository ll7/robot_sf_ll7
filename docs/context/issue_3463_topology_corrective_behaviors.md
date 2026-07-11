# Issue #3463 Topology Corrective Behaviors Integration Report

Issue: [#3463](https://github.com/ll7/robot_sf_ll7/issues/3463)

Status: diagnostic-only integration complete. All DoD criteria met after PR #5225 (2026-07-11). Issue #3465 remains the benchmark-facing promotion gate.

## Plain Summary

Issue #3463 converted the topology-guided local-planner lane from a revise-only
diagnostic into guarded, testable implementation surfaces. The CPU-only cross-slice
diagnostic ran 15 rows across three slices at `diagnostic_complete` strength. The
`doorway_transfer` blocker (scenario-level `obstacle_collision`) was removed from the
manifest in PR #4841. PR #5225 (2026-07-11) closed the last bounded corrective gap:
topology hypothesis availability now records valid/missing/malformed/configured-fallback
outcomes and fails closed on malformed geometry. This is not benchmark, planner-promotion,
paper, or dissertation evidence.

## Claim Boundary

This report consolidates merged implementation slices for Issue #3463 and maps
them to the issue acceptance criteria. The strongest supported claim is
diagnostic-only implementation readiness for topology candidate availability,
command arbitration, route-progress accounting, near-parity thresholds, and a
bounded sensitivity packet.

Broad benchmark improvement, planner promotion, paper-facing safety claims, and
paper-facing efficiency claims remain out of scope. The benchmark-facing
enabled-versus-disabled promotion gate remains
[#3465](https://github.com/ll7/robot_sf_ll7/issues/3465).

## Delivered Evidence

### Topology Hypothesis Availability

PR [#4388](https://github.com/ll7/robot_sf_ll7/pull/4388), merge
`85cc54750f660b92ecb82b4b04a2e0ec26ff0a05`, added episode-level
`topology_guided_episode` diagnostics in `robot_sf/benchmark/map_runner_episode.py`
with tests in `tests/benchmark/test_map_runner_utils.py`.

PR [#5225](https://github.com/ll7/robot_sf_ll7/pull/5225), merge
`3bca083f74a7170f1befff21458bb20af9c3af30`, closed the bounded corrective gap:
validates the topology command candidate's selected route geometry, records an
additive `topology_candidate_availability.v1` step diagnostic, aggregates
valid/missing/malformed/configured-fallback outcomes into the existing episode
diagnostic, and fails closed when the candidate source is malformed. Geometry-only
fallback summaries remain `fallback_only`. Merged 2026-07-11.

Status: covered by diagnostic metadata and corrective availability guard.

### Topology Candidate Influence

PR [#4426](https://github.com/ll7/robot_sf_ll7/pull/4426), merge
`3483ba35ff2a37abfb87e422f5695e001a58d2b0`, made `arbitration_weight` explicit,
added `blend_topology_command`, and records `topology_command_influence`.
PR [#4176](https://github.com/ll7/robot_sf_ll7/pull/4176), merge
`7291576e8ee321651dd75c38d6ca80ff94eb544c`, exposed topology corrective
controls.

Status: covered by code and focused tests.

### Fallback-Only Operation

`topology_guided_episode` records fallback status from PR #4388. Diagnostic
manifests and candidate registry rows keep `claim_boundary` and
`claim_eligibility` diagnostic-only. PR
[#4444](https://github.com/ll7/robot_sf_ll7/pull/4444), merge
`14147ee8b7b66b295e1167111e7ecc5acbd25ec3`, registers the Issue #3463
sensitivity packet as `diagnostic_only_not_benchmark_or_paper_evidence`.

Status: covered as a claim-boundary guard; not benchmark evidence.

### Route-Progress Accounting

PR [#4411](https://github.com/ll7/robot_sf_ll7/pull/4411), merge
`31dd50467ee9a7361cdc236bb4682358d713afde`, added
`topology_route_progress_state.v1` and near-parity churn metadata. PR #4444
registered the monotone progress-gated reselection candidate.

Status: covered by diagnostic metadata and focused tests; not benchmark evidence.

### Near-Parity Thresholds

PR #4411 added `topology_near_parity_thresholds.v1`. PR #4426 added nested
`topology_guided_hybrid_rule.v1` config and fail-closed validation for finite
non-negative thresholds and bounded arbitration weight.

Status: covered by code, config, and tests.

### Horizon And Scenario-Slice Sensitivity

PR #4444 registered `configs/policy_search/topology_reselection_cross_slice_issue_3463.yaml`,
candidate `topology_guided_hybrid_rule_v0_progress_gated_reselection_monotone`,
registry entries, and validator tests. PR
[#4600](https://github.com/ll7/robot_sf_ll7/pull/4600) preserved cross-slice
evidence. PR [#4746](https://github.com/ll7/robot_sf_ll7/pull/4746) preserved
blocker triage.

Status: executed at diagnostic-only strength; latest result is `blocked`, not
promotion evidence.

### Five Mechanism Families

The implemented surfaces cover topology hypothesis availability, command
arbitration strength, route-progress accounting, near-parity gate
parameterization, and horizon/scenario-slice sensitivity. The current unresolved
classification is the fail-closed `doorway_transfer` blocker in the 2026-07-05
and 2026-07-07 evidence.

Status: covered for diagnostic closure audit; blocker remains before promotion.

### Benchmark-Facing Gate

The issue thread records Issue #3465 as the successor gate. This report preserves
that boundary and does not close or supersede #3465.

Status: covered.

## Consolidated Contract

The #3463 contract is now complete and handed off to #3465:

- topology-guided corrective behavior remains diagnostic-only until #3465 or a
  successor explicitly runs the paired enabled-versus-disabled gate;
- fallback, degraded, missing-candidate, and fallback-only rows must not count
  as topology-lane success;
- topology-guided config fields stay finite, bounded, and documented through
  validation rules;
- Issue #3463 cross-slice diagnostic completed 15 rows at `diagnostic_complete`
  strength across three retained slices (doorway_transfer removed as scenario-level
  failure, not a topology bug);
- PR #5225 closes the topology hypothesis availability corrective gap.

## Remaining Blockers

| Blocker | Why it remains |
| --- | --- |
| No full benchmark promotion claim. | Issue #3463 intentionally stops at guarded diagnostic implementation; Issue #3465 owns the benchmark-facing enabled-versus-disabled comparison. |
| Two schema nitpicks from PR #4426 are tracked separately. | The issue thread spun off global-versus-per-step command limits and uniform topology-guided config schema handling to Issue #4430. |

## Next Empirical Action

The next empirical action for the topology lane is the benchmark-facing
enabled-versus-disabled comparison owned by Issue #3465. The cross-slice diagnostic
packet (manifest `configs/policy_search/topology_reselection_cross_slice_issue_3463.yaml`)
ran at diagnostic-only strength and is retained for reference. The `doorway_transfer`
slice remains in the evidence record as a scenario-level failure, preserved for future
behavioral work.
