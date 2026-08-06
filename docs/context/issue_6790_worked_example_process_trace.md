# Issue #6790 Worked-Example Process Trace

`worked_example_process_trace.v1` builds deterministic, renderer-neutral process diagnostics from
admitted `simulation_trace_export.v1` traces. It is analysis-workbench evidence only: it does not
change canonical benchmark metrics, SNQI, planner behavior, or figure admission.

## Public Surfaces

- Schema: `robot_sf/analysis_workbench/schemas/worked_example_process_trace.v1.json`
- Geometry-registry schema: `robot_sf/analysis_workbench/schemas/process_trace_geometry_registry.v1.json`
- Geometry-owner schema: `robot_sf/analysis_workbench/schemas/process_trace_geometry_owner.v1.json`
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
resolve to the same canonical record. Canonical encounter IDs are unique across the complete
report and an actor-prefixed ID must name its record actor. Without a report, every planner hint
in every frame and every `planner.encounters` list entry participates in focal binding;
contradictory actor or encounter IDs, including an actor-prefixed planner ID that names a different
actor, make the focal encounter explicitly unavailable.

Add `--pair-input path/to/other_trace.json --pair-comparison-grain ...` when building pair
compatibility; the comparison grain is required whenever a pair input is present.

Every analysis-affecting input is also recorded once in the versioned top-level
`analysis_input_contract`: the exact source digest, route and conflict registry inputs, pair
presence and full source receipt, report presence and full content, nullable focal selectors, and
the requested comparison grain. `analysis_input_sha256` is the canonical JSON digest of that
contract, and the full digest is part of `process_trace_id`. Semantic validation reconstructs the
complete artifact from the strict embedded `simulation_trace_export_receipt.v1` envelope and this
contract;
it does not trust the emitted focal actor, interval flags, frame indices, coordinate projections,
diagnostics, events, pair summaries, units, or coordinate-frame envelope. The embedded source
receipt contains a strict-JSON `content_contract` plus a canonical, sorted
`nonfinite_numbers` path/value ledger. Actual NaN and infinities become `null` only at the ledger's
RFC 6901 paths; decoding restores only validated ledger targets. A literal planner object that
resembles an old nonfinite marker remains ordinary content. The content SHA covers the entire
envelope, including the ledger, and the decoded contract must obey the exact source schema,
including its fixed evidence boundary and units and rejection of unknown envelope fields.

Content addressing inside an artifact detects partial rewrites, not a party that rewrites the
entire artifact and all of its self-authored hashes. Admission must therefore obtain the expected
SHA-256 of the exact official writer bytes independently and pass it as
`expected_artifact_sha256` to `validate_worked_example_process_trace`. Official bytes use
`indent=2`, sorted keys, `allow_nan=False`, UTF-8, and one trailing newline; the public
`serialize_worked_example_process_trace` and
`worked_example_process_trace_artifact_sha256` helpers define that contract. The CLI prints this
writer-byte SHA after writing. The external expected digest is the admission trust boundary; the
process-trace builder does not claim a signature or standalone authenticity.

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

`fixture_only` upstream bindings remain explicitly labeled in the emitted `owner_validation`
receipt and are available only as fixture proof. A `canonical_source` binding is stricter: the
owner reference must resolve in private validation context, its raw bytes must match the declared
SHA-256, and its selector must resolve exactly once to byte-semantically equal geometry in the
strict public `process_trace_geometry_owner.v1` envelope (`geometry_bindings` entries contain exact
`selector` and `geometry` objects). The owner loader rejects duplicate JSON keys and non-standard
NaN/Infinity constants and validates the complete envelope before scanning for a selector.
Missing, fabricated, digest-mismatched, malformed, ambiguous,
selector-mismatched, geometry-mismatched, or unrecognized owners make the projection explicitly
unavailable while preserving the supplied registry input receipt. The adapter therefore cannot
promote geometry merely because its own registry is internally consistent. Absolute paths,
including Windows drive-absolute forms on non-Windows hosts, remain private resolver context and
are rejected as public artifact references.

For canonical entries, pass repeatable `--geometry-owner REF=PATH` mappings to the CLI. `REF` is
the stable logical `source_artifact_ref`; `PATH` is private local resolver context threaded to both
route and conflict loaders and never emitted into the process trace. Malformed mappings and
duplicate logical references are rejected. Omitting a needed mapping does not silently bless the
adapter: unresolved canonical owners remain explicitly unavailable.

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
numeric timestamps. The event time stays the exact ledger time, while its step/frame is selected
deterministically as the first source sample at or after that time (within validated trace bounds),
never from whichever frame happened to carry the ledger record. Duplicate pedestrian IDs in any
left or pair-right source frame are rejected before focal lookup.

Pair compatibility requires an explicit comparison grain:

- `matched_planner_pair`: different planners on the same seed/realization, required
  initial-state equality, and equal map/horizon metadata. Planner config digests may differ.
- `matched_realization_pair`: the same planner/config on different seeds/realizations; start/spawn
  may differ and `shared_prefix=false` is allowed, with equal map/horizon/config metadata.

Event alignment selects the first available anchor from the declared fallback order and writes
`anchor_time_s` plus per-frame `tau_s = t - anchor_time_s`. The terminal fallback remains
unavailable because no canonical timed terminal-event contract exists for this issue; the builder
does not fabricate a terminal anchor from the final frame.
