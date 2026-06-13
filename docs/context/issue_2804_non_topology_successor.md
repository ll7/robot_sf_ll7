# Issue #2804 Non-Topology Successor Launch Packet (2026-06-13)

Issue: [#2804](https://github.com/ll7/robot_sf_ll7/issues/2804)
Parent: [#2801](https://github.com/ll7/robot_sf_ll7/issues/2801)
Successor to: [#2801](https://github.com/ll7/robot_sf_ll7/issues/2801) (non-topology follow-up)
Predecessor chain: [#2742](https://github.com/ll7/robot_sf_ll7/issues/2742) → [#2751](https://github.com/ll7/robot_sf_ll7/issues/2751) → [#2752](https://github.com/ll7/robot_sf_ll7/issues/2752) → [#2801](https://github.com/ll7/robot_sf_ll7/issues/2801)

Claim boundary: `analysis_only_not_benchmark_or_paper_evidence`.

## Summary

Analysis-only launch packet for the **local_policy_scoring** investigation of the
`t_intersection_transfer` hard slice. This is the non-topology successor chosen by #2801 after
stopping topology-reselection-as-clearance for the current hard-slice set.

The t_intersection_transfer slice (classic_t_intersection_medium) showed 13 topology switches
across all progress-gated thresholds (0.05, 0.1, 0.2) at identical 6.39m route progress, still
`horizon_exhausted`. The mechanism diagnosis (#2752) classified this as
`mechanism_or_scenario_ambiguous`: switching helped reach further than baseline (4.87m → 6.39m)
but cannot distinguish `blocked_geometry` from `switch_too_often` without per-step
`switch_timeline` evidence.

## Selected Intervention Target

**local_policy_scoring** for `t_intersection_transfer` to separate `blocked_geometry` from
`switch_too_often` using per-step scoring diagnostics.

## Hypothesis

Local-policy scoring diagnostics on the t_intersection_transfer trace will reveal whether the
13 topology switches at 6.39m progress are caused by:

1. **blocked_geometry**: the local policy repeatedly scores the same blocked route as best because
   no viable alternative geometry exists within the scoring horizon, or
2. **switch_too_often**: the local policy oscillates between candidates due to scoring noise or
   insufficient hysteresis, preventing stable progress on any single route.

## Comparator / Baseline

- **Baseline**: topology-guided hybrid rule v0 with progress-gated reselection at threshold 0.05m
  (the most sensitive progress-gated row from #2751), which produced 6.39m progress with 13
  switches and `horizon_exhausted`.
- **Comparator**: per-step local-policy scoring trace on the same seed/scenario, recording
  candidate scores, chosen candidate, and switch events at every planning step.

## Stop Rule

**Stop this investigation if:**

1. Per-step scoring trace confirms that all candidates receive near-identical low scores throughout
   the episode. Predeclared threshold: in at least 80% of scoring steps, the best-vs-second-best
   candidate score gap is below 5% of the per-episode absolute score range and route progress
   remains below 6.5m. This indicates `blocked_geometry` is the dominant mechanism and no
   local-policy scoring refinement can clear the slice without scenario geometry changes.
2. Per-step scoring trace shows rapid oscillation (>10 switches in <20% of episode steps) with no
   candidate maintaining best-score for at least five consecutive scoring steps, indicating
   `switch_too_often` and pointing to a hysteresis/stability fix rather than a scoring-model
   change.
3. The per-step scoring trace cannot be produced from the existing runner or adapter without
   non-trivial code changes; in that case, close as `blocked` with the exact instrumentation gap.

**Continue to a runtime packet if:**

- Per-step scoring trace shows a clear scoring signal that differentiates at least one candidate
  as consistently best but the current selection mechanism fails to lock onto it, indicating a
  fixable selection/hysteresis gap.

## Hard Slices

| Slice | Scenario | Role |
|---|---|---|
| t_intersection_transfer | classic_t_intersection_medium | hard |

## Negative Control

Reuse the `simple_negative_control` (empty_map_8_directions_east) from #2751. Expect zero topology
switches and stable best-candidate scoring throughout the episode, confirming the per-step scoring
instrumentation does not introduce spurious switching or scoring artifacts.

## Decision Rule

- **If blocked_geometry confirmed**: close the slice as `stop` for local-policy scoring; the
  required fix is scenario geometry redesign (wider intersection, different obstacle placement,
  or additional route alternatives). Update the negative result register.
- **If switch_too_often confirmed**: close as `revise` with a concrete hysteresis or stability
  recommendation (e.g., minimum-hold-steps, score-delta threshold, or candidate-reuse penalty).
  This would be a mechanism fix, not a benchmark claim.
- **If clear scoring signal exists but selection fails**: close as `revise` with a specific
  selection-mechanism fix recommendation. This could support a future runtime packet.
- **If instrumentation blocked**: close as `blocked` with the exact gap named.

## Artifact Plan

No runtime artifacts are produced by this launch packet. The durable artifacts are:

- This context note: `docs/context/issue_2804_non_topology_successor.md`
- Evidence summary: `docs/context/evidence/issue_2804_non_topology_successor/summary.json`
- Evidence README: `docs/context/evidence/issue_2804_non_topology_successor/README.md`
- If a runtime investigation is later launched, its artifacts belong in a separate evidence
  directory under `docs/context/evidence/` with its own summary and claim boundary.

## Why This Follows From #2801

Issue #2801 recommended stopping topology-reselection for the current hard-slice set and named
two non-topology successor targets:

1. `scenario_design_or_geometry` for doorway/bottleneck (high/medium confidence scenario
   insufficiency).
2. `local_policy_scoring` for t_intersection (mechanism_or_scenario_ambiguous).

This packet implements option 2. It does **not** promote #2751 to benchmark or paper-facing
evidence. The #2751 runtime evidence remains `diagnostic_only_not_benchmark_or_paper_evidence`,
and this launch packet inherits that claim boundary. The per-step scoring investigation is a
mechanism-diagnostic step, not a clearance benchmark.

## Caveats

- This is analysis-only. It does not prove that local-policy scoring can clear the t_intersection
  slice; it only proposes a diagnostic to separate two candidate failure mechanisms.
- The doorway/bottleneck slices remain classified as scenario insufficiency; this packet does not
  address them. A separate scenario_design_or_geometry investigation would be needed.
- If the per-step scoring trace requires new instrumentation in the topology-hypothesis runner or
  the local-policy adapter, that instrumentation work should be scoped separately and named as a
  blocker if it exceeds a small change.
- This packet does not rerun #2751 unchanged. It proposes a new diagnostic surface (per-step
  scoring) that was not available in the #2751 runtime evidence.

## Related

- Issue #2801: topology successor recommendation (stop decision)
- Issue #2751: runtime evidence for clearance-targeted reselection
- Issue #2752: mechanism diagnosis of hard-slice failures
- Issue #2742: topology reselection successor launch packet
- Issue #2716: cross-slice topology reselection context
- Evidence bundle: `docs/context/evidence/issue_2804_non_topology_successor/`
