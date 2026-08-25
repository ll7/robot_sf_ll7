# Route-Side and Homotopy Observability Contract (`route_choice_observability`)

**Status:** diagnostic / analysis-only — metric semantics and deterministic fixtures, not evidence
of pedestrian preference, response, comfort, or general human predictability.
**Issue:** [#7890](https://github.com/ll7/robot_sf_ll7/issues/7890) (parent research
[#7883](https://github.com/ll7/robot_sf_ll7/issues/7883)).
**Owner module:** `robot_sf/benchmark/route_choice_observability.py`.
**Topology source:** `scripts/validation/run_topology_hypothesis_diagnostics.py`
(`_RouteHypothesisPath`, `_topology_signature`), `docs/context/issue_1692_topology_hypothesis_probe.md`,
`docs/context/issue_1674_topology_hypothesis_diagnostics.md`.

Plain-language summary: this contract describes, deterministically and frame-explicitly, which side
and which route/corridor hypothesis a planned path uses. It separates route-side classification,
topological/homotopy identity, temporal consistency across replans, and unavailable-or-ambiguous
observations. It measures **planner-route observability** only — never human predictability.

## 1. Reference frame

Route side is defined relative to a declared directed reference axis from scenario start to goal.
The record carries:

- coordinate frame (`global_xy` by default);
- start and goal/reference points;
- units (metres);
- numerical tolerance (`tolerance_m`, default 0.05) and neutral-band half-width
  (`neutral_band_m`, default 0.2);
- the path progress interval used for classification (default `(0.1, 0.9)`).

A left/right label without this reference is invalid — the classifier fails closed with
`degenerate_reference` when the start-goal axis has zero length.

## 2. Route-side vocabulary

`left` / `right` / `neutral` / `mixed` / `unavailable`.  The left-hand side is the clockwise
perpendicular of the directed axis (facing the goal).  A path that traverses both strict sides is
`mixed` — never whichever side has the last sample.  Missing, non-finite, zero-length,
single-point, or degenerate geometry fails closed as `unavailable` with an explicit reason
(`empty_path`, `single_point`, `zero_length`, `non_finite`, `degenerate_reference`,
`insufficient_progress`).

## 3. Homotopy identity

Reuses the compact corridor-signature idea from the topology-hypothesis diagnostics: choke cells of
the path relative to the blocked map.  The identity is a canonical, order-independent string of
choke cells (`row,col` pairs sorted and joined).  It is stable across discovery order and does not
depend on ephemeral route names such as `primary_route` or `masked_cell_*`.

## 4. Temporal consistency

For a sequence of replanned paths, the report records valid/unavailable counts separately,
side-transition and topology-transition counts, dominant side/topology, consistency fraction with
an explicit denominator, and the first stable-decision step when defined.  Outputs are never merged
into a single social-compliance score.

## 5. Claim boundary

Collision, near-miss, minimum clearance, path length, and smoothness remain separate existing
metrics; this contract does not alter their semantics.  The receipt makes no claim about pedestrian
response, social preference, or general human predictability — completion proves route-choice
observability contract validity only.
