# Issue #2801 Topology Reselection Successor Recommendation (2026-06-13)

Issue: [#2801](https://github.com/ll7/robot_sf_ll7/issues/2801)
Parent: [#2742](https://github.com/ll7/robot_sf_ll7/issues/2742)
Successor to: [#2751](https://github.com/ll7/robot_sf_ll7/issues/2751), [#2752](https://github.com/ll7/robot_sf_ll7/issues/2752)

Claim boundary: `analysis_only_not_benchmark_or_paper_evidence`.

## Summary

Analysis-only successor recommendation after consuming #2751 runtime evidence and #2752 mechanism
diagnosis. The recommendation is **stop** for topology-reselection-as-clearance on the current
hard-slice set. Two of three hard slices are scenario/geometry insufficiency; the third is
ambiguous but switching did not produce clearance. No further topology-reselection parameter
tuning is expected to clear these slices.

## Mechanism Diagnosis Used

Source: `docs/context/issue_2752_topology_reselection_mechanism.md`

| Slice | Primary Label | Confidence | Mechanism vs Scenario |
|---|---|---|---|
| bottleneck_transfer | no_useful_topology_alternative | medium | scenario_insufficiency_likely |
| doorway_transfer | no_useful_topology_alternative | high | scenario_insufficiency |
| t_intersection_transfer | candidate_route_blocked | medium | mechanism_or_scenario_ambiguous |

## Selected Intervention Target

**None for topology reselection.** The mechanism diagnosis shows that topology reselection
parameters (reuse_penalty, progress_gated thresholds 0.05/0.1/0.2) are not the limiting factor:

- doorway_transfer: all five candidates produced identical 1.45m progress with 0 switches.
- bottleneck_transfer: progress-gated rows show 0 switches; reuse_penalty gained ~0.9m but
  remained horizon_exhausted, indicating geometry is the dominant constraint.
- t_intersection_transfer: switching increased progress from 4.87m to 6.39m but 13 switches
  at the same final progress cannot distinguish blocked_geometry from switch_too_often.

The recommended next actions are outside the topology-reselection scope:

1. **scenario_design_or_geometry** for doorway_transfer and bottleneck_transfer (high/medium
   confidence scenario insufficiency).
2. **local_policy_scoring** for t_intersection_transfer to separate blocked_geometry from
   switch_too_often using per-step switch_timeline.

The dedicated non-topology follow-up is
[#2804](https://github.com/ll7/robot_sf_ll7/issues/2804).

## Rejected Alternatives

| Alternative | Reason Rejected |
|---|---|
| Further progress-gated threshold tuning | Three thresholds (0.05, 0.1, 0.2) produced identical outcomes per slice; gate parameter is not the limiting factor. |
| Topology-reselection runtime packet | No hard slice cleared; mechanism diagnosis shows scenario insufficiency for 2/3 slices, ambiguity for the third. A runtime packet would reproduce the same horizon_exhausted outcome. |
| Reuse-penalty refinement | reuse_penalty gained marginal progress on bottleneck_transfer (6.1m vs 5.2m) but remained horizon_exhausted; no gain on doorway_transfer. |
| New topology-reselection candidate config | The mechanism shows no_useful_topology_alternative for 2/3 slices; a new candidate config would face the same geometry constraint. |

## Hypothesis (If Runtime Packet Were Proposed)

Not proposed. The mechanism diagnosis provides sufficient evidence that topology reselection
is not the clearance bottleneck for the current hard-slice set.

## Comparator (If Runtime Packet Were Proposed)

Not applicable. No runtime packet is proposed.

## Stop Rule

**Stop topology-reselection-as-clearance for the current hard-slice set.** The stop condition
is met because:

1. All three hard slices remained `horizon_exhausted` across all candidates.
2. Mechanism diagnosis classifies 2/3 as scenario insufficiency (one high confidence).
3. The third slice shows switching occurred but did not produce clearance, and the
   distinguishing evidence (per-step switch_timeline) is not in durable git evidence.
4. Negative control succeeded with zero switching, confirming the mechanism does not
   introduce spurious switches.

## Artifact Plan

No runtime artifacts are produced. The durable artifacts are:

- This context note: `docs/context/issue_2801_topology_successor_recommendation.md`
- Evidence README: `docs/context/evidence/issue_2801_topology_successor_recommendation/README.md`
- Evidence summary: `docs/context/evidence/issue_2801_topology_successor_recommendation/summary.json`
- Non-topology follow-up: [#2804](https://github.com/ll7/robot_sf_ll7/issues/2804)

## Caveats

- This is analysis-only evidence. It does not prove that topology reselection cannot work
  on any scenario; it only recommends stopping for the current hard-slice set
  (classic_bottleneck_medium, classic_doorway_medium, classic_t_intersection_medium).
- t_intersection_transfer remains ambiguous; a local_policy_scoring investigation could
  separate mechanism from scenario if a maintainer decides the investment is worthwhile.
- Scenario design changes could create new hard slices where topology reselection does
  provide clearance. Those would need their own launch packet and successor evaluation.
- Follow-up #2804 is intentionally non-topology scoped; it should pick either scenario geometry
  design or local-policy scoring before any new runtime packet is launched.

## Related

- Issue #2742: topology reselection successor packet
- Issue #2751: runtime evidence for clearance-targeted reselection
- Issue #2752: mechanism diagnosis of hard-slice failures
- Issue #2804: non-topology successor follow-up
- Issue #2716: cross-slice topology reselection context
- Evidence bundle: `docs/context/evidence/issue_2801_topology_successor_recommendation/`
