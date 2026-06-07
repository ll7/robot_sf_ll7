# Issue #2452 Mechanism-Aware Local-Navigation Suites

Issue: [#2452](https://github.com/ll7/robot_sf_ll7/issues/2452)
Status: proposal manifest; not benchmark evidence.

## Purpose

`configs/benchmarks/issue_2452_mechanism_aware_local_nav_suites_v0.yaml` defines the first
mechanism-aware local-navigation suite registry. The registry groups existing scenario IDs into
small suites so follow-up benchmark work can test why a planner succeeds or fails rather than only
whether aggregate success, collision, or timeout changed.

## Claim Boundary

This is launch infrastructure only. It does not establish a benchmark result, planner ranking,
transfer claim, AMV-realism claim, publication panel, or paper-facing conclusion. Fallback,
degraded, failed, missing, partial, and not-available rows remain visible row-status outcomes and
must not be counted as suite-strengthening evidence.

## Suite Registry

| Suite | Target mechanism | Scenario IDs | Minimum tier |
| --- | --- | --- | --- |
| `static_deadlock_recovery` | `static_deadlock_or_local_minimum` | `classic_bottleneck_low`, `classic_head_on_corridor_low`, `narrow_passage` | `controlled_trace` |
| `topology_hypothesis_selection` | `route_or_topology_mismatch` | `classic_realworld_double_bottleneck_high`, `classic_t_intersection_medium`, `symmetry_ambiguous_choice` | `controlled_trace` |
| `dynamic_phase_sensitivity` | `dynamic_phase_or_order_sensitivity` | `francis2023_intersection_wait`, `francis2023_join_group`, `classic_merging_low` | `controlled_trace` |
| `proxemic_tradeoff` | `proxemic_or_clearance_tradeoff` | `classic_group_crossing_high`, `classic_overtaking_medium`, `classic_t_intersection_medium` | `controlled_trace` |
| `actuation_feasibility` | `actuation_or_command_saturation` | `classic_head_on_corridor_low`, `classic_overtaking_medium`, `classic_bottleneck_high` | `controlled_trace` |
| `guard_domination` | `guard_or_handoff_domination` | `classic_crossing_low`, `classic_bottleneck_low`, `classic_head_on_corridor_low` | `diagnostic_smoke` |
| `learned_low_progress` | `learned_policy_low_progress` | `classic_cross_trap_low`, `classic_group_crossing_high`, `classic_head_on_corridor_low` | `diagnostic_smoke` |

Every suite declares candidate baselines, candidate interventions, required trace fields, required
metrics, a seed set, and a local claim boundary. The manifest intentionally names candidates, not
validated executable rows.

## Source Alignment

- [Issue #2220 Failure-Mechanism Taxonomy](issue_2220_failure_mechanism_taxonomy.md) supplies the
  target mechanism vocabulary.
- [Issue #2232 Planner Mechanism Transfer Benchmark Protocol](issue_2232_planner_mechanism_transfer_benchmark.md)
  supplies the transfer-evidence boundary and prevents transfer claims from this proposal alone.
- [Issue #2389 Mechanism-Aware Evaluation Thread](issue_2389_mechanism_aware_evaluation_thread.md)
  supplies the current stop, revise, and blocked research directions.
- [Issue #2447](https://github.com/ll7/robot_sf_ll7/issues/2447) requires nonzero mechanism signal
  before mechanism-panel publication.

## Next Proof Step

Pick exactly one suite and bind it to executable configs, native or explicitly eligible adapter
planner rows, and durable row-status artifacts. The first follow-up should require the manifest's
trace fields to be emitted before interpreting metric deltas. If those fields are unavailable, the
row should fail closed or revise the suite contract instead of claiming benchmark evidence.

## Validation

Proposal validation should verify the manifest contract and referenced paths:

```bash
uv run pytest tests/benchmark/test_issue_2452_mechanism_aware_suites.py -q
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
git diff --check
```
