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
- the path progress interval used for classification (default `(0.1, 0.9)`), measured as
  normalized cumulative polyline arc length rather than sample index.

A left/right label without this reference is invalid — the classifier fails closed with
`degenerate_reference` when the start-goal axis has zero length.

## 2. Route-side vocabulary

`left` / `right` / `neutral` / `mixed` / `unavailable`. The left-hand side is the standard
counter-clockwise perpendicular of the directed axis (facing the goal): +Y is left for a +X
reference direction in `global_xy`. A path that traverses both strict sides is
`mixed` — never whichever side has the last sample.  Missing, non-finite, zero-length,
single-point, or degenerate geometry fails closed as `unavailable` with an explicit reason
(`empty_path`, `single_point`, `zero_length`, `non_finite`, `degenerate_reference`,
`insufficient_progress`, `invalid_tolerance`, `invalid_neutral_band`,
`invalid_progress_interval`, `invalid_reference`, `invalid_path`).

Classification clips every intersecting polyline segment to the declared arc-length interval and
evaluates signed-distance extrema. Repeating points or inserting samples along existing segments
therefore cannot change a label.

## 3. Homotopy identity

Reuses the canonical corridor-signature helper from the topology-hypothesis diagnostics: choke
cells of the path relative to the blocked map, with the same finite-clearance fallback when no
choke cell exists. The identity is a canonical, order-independent string of `row,col` pairs sorted
and joined. It is stable across discovery order and does not depend on ephemeral route names such
as `primary_route` or `masked_cell_*`. Malformed maps and invalid thresholds fail closed.
Path coordinates must be integral grid cells, stay in bounds, and avoid blocked cells; continuous
world coordinates must be converted by the map owner before this helper is called.

## 4. Temporal consistency

For a sequence of replanned paths, the report records valid/unavailable counts separately,
side-transition and topology-transition counts, dominant side/topology, consistency fraction with
explicit side, topology, and aligned denominators, and the first stable-decision step when
defined. Side/topology transitions do not bridge unavailable samples, and length-mismatched
sequences are returned as alignment-invalid with zero admissible denominator. Outputs are never
merged into a single social-compliance score.

`consistency_fraction` is the modal aligned `(side, topology)` pair count divided by the aligned
valid-pair count. `availability_fraction` separately reports aligned valid pairs divided by all
aligned replanning steps. `first_stable_step` is the first step beginning an uninterrupted suffix
of at least two identical valid `(side, topology)` pairs; an unavailable sample breaks stability.

## 5. Versioned diagnostic record

`diagnostic_record(...)` emits `route_choice_observability.v1`, including the side and homotopy
observations, temporal report, `analysis-only` evidence tier, `diagnostic-only` result
classification, and the claim boundary below. An alignment mismatch or a sequence without any
valid aligned pair emits `status: not_available`.

The representative deterministic output is tracked in
[`evidence/issue_7890_route_choice_observability/receipt.json`](evidence/issue_7890_route_choice_observability/receipt.json).

## 6. Claim boundary

Collision, near-miss, minimum clearance, path length, and smoothness remain separate existing
metrics; this contract does not alter their semantics.  The receipt makes no claim about pedestrian
response, social preference, or general human predictability — completion proves route-choice
observability contract validity only.
