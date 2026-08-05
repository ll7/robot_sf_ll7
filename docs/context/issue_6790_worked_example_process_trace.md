# Issue #6790 Worked-Example Process Trace

`worked_example_process_trace.v1` builds deterministic, renderer-neutral process diagnostics from
admitted `simulation_trace_export.v1` traces. It is analysis-workbench evidence only: it does not
change canonical benchmark metrics, SNQI, planner behavior, or figure admission.

## Public Surfaces

- Schema: `robot_sf/analysis_workbench/schemas/worked_example_process_trace.v1.json`
- Geometry-registry schema: `robot_sf/analysis_workbench/schemas/process_trace_geometry_registry.v1.json`
- Builder and coordinate orchestration: `robot_sf/analysis_workbench/interaction_coordinates.py`
- Pair compatibility: `robot_sf/analysis_workbench/event_alignment.py`
- Stall/reversal phase summaries: `robot_sf/analysis_workbench/episode_phases.py`
- Proxy clearance and TCPA/CPA diagnostics: `robot_sf/analysis_workbench/safety_surrogates.py`
- CLI: `scripts/analysis/build_worked_example_process_trace.py`
- Tests: `tests/analysis_workbench/test_worked_example_process_trace.py`

## Reproduction Command

```bash
uv run python scripts/analysis/build_worked_example_process_trace.py \
  --input tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json \
  --geometry-registry tests/fixtures/analysis_workbench/process_trace_geometry_registry_v1/fixture_registry.json \
  --route-entry-id fixture-route \
  --conflict-zone-entry-id fixture-zone \
  --out output/worked_example_process_trace_fixture.json
```

Add `--encounter-report path/to/near_miss_encounter.v1.json` when binding the focal actor
and interval to canonical near-miss encounter output. The report must validate against the
canonical `near_miss_encounter.v1` schema and its provenance `input_checksums` must include
the SHA-256 of the input trace file.
The focal record embeds a strict-JSON report content contract plus report and selected-entry
SHA-256 receipts. Validation rechecks the canonical report schema and requires the declared actor,
encounter, and complete selected record to replay that receipt exactly.
Use `--focal-actor-id` to select an actor across every canonical report encounter, or
`--focal-encounter-id` to select one unique encounter directly. When both are present they must
resolve to the same canonical record.

Add `--pair-input path/to/other_trace.json --pair-comparison-grain ...` when building pair
compatibility; the comparison grain is required whenever a pair input is present.

The analysis workbench owns `process_trace_geometry_registry.v1` as an adapter and receipt format,
not as a second production map-authoring source. Production registry entries must derive from or
explicitly bind the hash-pinned owner artifact and its native selector. Routes should preserve
available `map_id`, `spawn_id`, `goal_id`, and SVG/`GlobalRoute` identity; conflict zones should
preserve their scenario/map zone identity. Entries that do not have such an owner, including this
contract fixture, must say `kind: fixture_only` and must not imply that an arbitrary `route_id` is
canonical.

Route and conflict projections are available only after the builder loads a unique entry from an
actual world-frame registry JSON file. Output receipts bind a stable repo-relative or logical
`artifact_ref` (never an absolute checkout path), raw file SHA-256, registry and entry IDs,
canonical entry SHA-256, upstream binding, coordinate frame, geometry kind, and resolved
coordinates. Semantic validation resolves that reference through local validator context, reopens
the file, and replays the entry instead of trusting caller geometry or emitted receipt strings.
This keeps identical bytes content-identical across checkout roots while still failing closed for
missing, moved, tampered, duplicate, non-world, or ambiguous evidence.
Each route and conflict contract also declares its input independently as `not_supplied`,
`supplied` with the registry artifact/content/entry digests, or `supplied_unregistered` for an
explicit caller input that cannot claim an external receipt. Top-level and per-frame
availability replay from that declaration and the source coordinate frame; pair-right events bind
the same input declaration. A projection-unavailable source frame therefore cannot hide a corrupt
supplied registry receipt.

Ordered route polylines use cumulative arclength and nearest-segment projection; equal-distance
ties abstain as ambiguous. Zero-length, adjacent backtracking/overlap, and nonlocal intersections
also fail closed. `route_graph` is schema-recognized only so branched authoring inputs can
fail closed explicitly; it is not projectable. Conflict projection currently supports circles
only. Point or polygon zone owners remain explicitly unavailable until a versioned projection
contract is added. The CLI intentionally has no direct geometry/checksum arguments.

Canonical timed ledger collisions remain episode-level exact-collision anchors for pedestrian and
static-geometry partners. The deterministic earliest canonical episode collision remains selected;
focal matching is metadata and never replaces it with a later collision. Partner type and ID remain
truthful, while `actor_id` is null unless a separate `focal_binding` record proves that the
pedestrian partner matches the selected focal encounter. Boolean values are never admitted as
numeric timestamps. Duplicate pedestrian IDs in any left or pair-right source frame are rejected
before focal lookup.

Pair compatibility requires an explicit comparison grain:

- `matched_planner_pair`: different planners on the same seed/realization, required
  initial-state equality, and equal map/horizon metadata. Planner config digests may differ.
- `matched_realization_pair`: the same planner/config on different seeds/realizations; start/spawn
  may differ and `shared_prefix=false` is allowed, with equal map/horizon/config metadata.

Event alignment selects the first available anchor from the declared fallback order and writes
`anchor_time_s` plus per-frame `tau_s = t - anchor_time_s`. The terminal fallback remains
unavailable because no canonical timed terminal-event contract exists for this issue; the builder
does not fabricate a terminal anchor from the final frame.
