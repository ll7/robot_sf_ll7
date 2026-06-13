# Issue #2752 Topology Reselection Mechanism Diagnosis (2026-06-13)

Issue: [#2752](https://github.com/ll7/robot_sf_ll7/issues/2752)
Parent: [#2742](https://github.com/ll7/robot_sf_ll7/issues/2742)
Successor to: [#2751](https://github.com/ll7/robot_sf_ll7/issues/2751)

Claim boundary: `analysis_only_not_benchmark_or_paper_evidence`.

## Summary

Mechanism-level failure classification of three hard slices from Issue #2751 runtime evidence.
All hard slices remained `horizon_exhausted` after topology reselection; no topology alternative
produced clearance. Two slices are likely scenario/geometry insufficiency; one is ambiguous
between blocked geometry and excessive switching.

## Failure Labels

| Slice | Primary Label | Confidence | Mechanism vs Scenario | Next Action |
|---|---|---|---|---|
| bottleneck_transfer | no_useful_topology_alternative | medium | scenario_insufficiency_likely | scenario_design_or_geometry |
| doorway_transfer | no_useful_topology_alternative | high | scenario_insufficiency | scenario_design_or_geometry |
| t_intersection_transfer | candidate_route_blocked | medium | mechanism_or_scenario_ambiguous | local_policy_scoring |

## Evidence

- Runtime evidence: `docs/context/evidence/issue_2751_topology_reselection_runtime/`
- Mechanism evidence bundle: `docs/context/evidence/issue_2752_topology_reselection_mechanism/`

## Key Findings

1. **doorway_transfer** (high confidence): All five candidates produced identical behavior at
   1.45m progress with 0 switches and 159 deadlock steps. This is the cleanest
   `no_useful_topology_alternative` signal; no topology variant changed behavior.

2. **bottleneck_transfer** (medium confidence): All candidates `horizon_exhausted` at ~5.2-6.1m.
   Progress-gated rows show 0 switches. `reuse_penalty` reached 6.1m with 2 switches but still
   `horizon_exhausted`, suggesting the geometry bottleneck is the dominant constraint.

3. **t_intersection_transfer** (medium confidence): `reuse_penalty` and all progress-gated
   thresholds reached 6.39m with 13 switches but still `horizon_exhausted`. Baseline achieved
   4.87m with 3 switches. Switching helped reach further but could not clear the intersection.
   Cannot distinguish `blocked_geometry` from `switch_too_often` without per-step
   `switch_timeline`.

## Diagnostic Gaps

- Per-step `switch_timeline` exists only in worktree-local `topology_hypotheses.json` traces,
  not in durable git evidence.
- No automated mechanism-diagnosis script exists.
- Progress-gated thresholds produced identical outcomes; gate parameter is not the limiting
  factor for these scenarios.

## Negative Control

`simple_negative_control` succeeded with zero topology switching across all candidates,
confirming the mechanism does not introduce spurious switching in trivial scenarios.

## Related

- Issue #2742: topology reselection successor packet
- Issue #2751: runtime evidence for clearance-targeted reselection
- Issue #2716: cross-slice topology reselection context
- Evidence bundle: `docs/context/evidence/issue_2752_topology_reselection_mechanism/`
