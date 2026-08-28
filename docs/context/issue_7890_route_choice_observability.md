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

Strict left/right evidence begins beyond `neutral_band_m + tolerance_m`; the tolerance therefore
widens the neutral boundary instead of serving only as recorded metadata. Reference points must be
exactly two-dimensional, and coordinate-frame and units labels must be non-empty.

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
choke cell exists. The identity is a canonical, order-independent string of coordinate pairs sorted
and joined. Pure grid callers default to `occupancy_grid_rc` coordinates in cells. The
topology-guided planner computes the observation against its own selected path and blocked map and
emits the aligned `global_xy` choke-point set. Temporal reporting uses strict bounded symmetric
Hausdorff matching with the planner grid resolution as tolerance and deterministic complete-link
clusters, so every pair in one reported topology cluster matches. Moving cell representatives less
than one cell apart remain one topology while corridors a full cell apart remain distinct.
Every observation records `identity_coordinate_frame`, `identity_units`, `identity_points`, and
`identity_match_tolerance`; changing any reference field invalidates a temporal comparison.

Identity remains stable across discovery order and does not depend on ephemeral route names such as
`primary_route` or `masked_cell_*`. The production diagnostic consumes the planner's exact
`route_path_grid`, `route_path_world`, and `route_homotopy_observation` payload; it does not
regenerate a blocked map or rejoin alternatives by those names. Malformed maps, invalid thresholds
or match tolerances, misaligned coordinates, and ambiguous frame metadata fail closed. Grid paths must
be integral, stay in bounds, avoid blocked cells, and use duplicate or 8-connected consecutive
steps; non-adjacent jumps fail closed.

Only a topology-guided planner decision that exposes this exact selected path can produce an
operational homotopy observation. Base route-corridor diagnostics without selected-topology path
provenance remain explicitly `unavailable`.

## 4. Temporal consistency

For a sequence of replanned paths, the report records valid/unavailable counts separately,
side-transition and topology-transition counts, dominant side/topology, consistency fraction with
explicit side, topology, and aligned denominators, and the first stable-decision step when
defined. Side/topology transitions do not bridge unavailable samples, and length-mismatched
sequences are returned as alignment-invalid with zero admissible denominator. A change in the
declared route reference, classification thresholds, homotopy coordinate frame, homotopy units, or
match tolerance likewise fails alignment closed rather than comparing incompatible labels. Outputs
are never merged into a single social-compliance score.

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
