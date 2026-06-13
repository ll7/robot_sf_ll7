# Issue #2752 Topology Reselection Mechanism Diagnosis (2026-06-13)

Claim boundary: `analysis_only_not_benchmark_or_paper_evidence`.
Classification: `analysis_only`.

Mechanism-level failure classification of three hard slices from Issue #2751 runtime evidence.
All hard slices remained `horizon_exhausted`; no topology alternative produced clearance.

## Decision Table

| Slice | Primary Failure Label | Confidence | Evidence Summary | Mechanism vs Scenario | Next Action |
|---|---|---|---|---|---|
| bottleneck_transfer | no_useful_topology_alternative | medium | All candidates horizon_exhausted ~5.2-6.1m. Progress-gated: 0 switches. reuse_penalty: 2 switches, 6.1m, still stuck. | scenario_insufficiency_likely | scenario_design_or_geometry |
| doorway_transfer | no_useful_topology_alternative | high | All candidates identical at 1.45m, 0 switches, 159 deadlock steps. No topology variant changed behavior. | scenario_insufficiency | scenario_design_or_geometry |
| t_intersection_transfer | candidate_route_blocked | medium | reuse_penalty/progress_gated: 6.39m, 13 switches, still horizon_exhausted. Baseline: 4.87m, 3 switches. | mechanism_or_scenario_ambiguous | local_policy_scoring |

## Negative Control

`simple_negative_control` succeeded with zero topology switching across all candidates, confirming
the mechanism does not introduce spurious switching in trivial scenarios.

## Caveats

- Evidence is analysis-only; no benchmark or paper-facing claims.
- Per-step `switch_timeline` data exists only in worktree-local `topology_hypotheses.json` traces,
  not in durable git evidence. This prevents separating `switch_too_often` from
  `candidate_route_blocked` for `t_intersection_transfer`.
- Progress-gated thresholds (0.05, 0.1, 0.2) produced identical outcomes per slice, so the gate
  parameter is not the limiting factor for these scenarios.
- No mechanism-diagnosis script exists; classification was manual from `summary.json` and scout
  analysis.
- `bottleneck_transfer`: reuse_penalty reached slightly more progress (6.1m vs 5.2m) with 2
  switches but remained `horizon_exhausted`, suggesting the geometry bottleneck dominates.
- `t_intersection_transfer`: switching helped reach further (6.39m vs 4.87m baseline) but still
  could not clear the intersection; 13 switches at identical final progress could indicate
  oscillation between equally-bad alternatives.

## Diagnostic Gaps

1. **Missing per-step switch_timeline**: Cannot distinguish whether `t_intersection_transfer`
   failures are due to blocked geometry or excessive switching between equally-poor candidates.
2. **No automated classification script**: Manual classification from summary-level metrics; a
   small helper script could automate this for future runs.
3. **Progress-gated parameter not limiting**: All thresholds produced identical outcomes, suggesting
   the reselection gate is not the constraint for these scenario geometries.
