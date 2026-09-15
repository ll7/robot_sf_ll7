# Review storyboard (SREV-07)

Command-line glossary: SREV is the scenario-review portfolio; a storyboard
spec (`visualization-spec.v1`) is the ranked-candidate document with source
intervals; a component request (`component-request.v1`) invokes one workbench
component; shared contract shapes live in
`robot_sf/analysis_workbench/review_contracts.py` (SREV-01).

## What it does

`python -m robot_sf.analysis_workbench.review_storyboard` ranks review-bundle
episodes with deterministic tie-breaking (score descending, then episode ID
ascending). The emitted spec binds each selected episode to the actual bundle
reference artifact IDs. A declared override source digest must match the
current bundle before pins or exclusions apply; otherwise overrides are
ignored with an explicit diagnostic. Event intervals are clipped to the
configured source duration, while their interval IDs and source row identity
remain in the annotation payload next to the renderer-neutral time intervals.

The component accepts either the canonical
`trace_exemplar_interest` report shape (`episodes` entries with
`composite_score`) through the explicitly versioned adapter
`trace-exemplar-interest.report-adapter.v1`, or the explicitly versioned
`srev07-exemplar-scores.v1` score map. The canonical adapter requires the full
owner report envelope (`roots`, `weights`, `episodes`, and
`comparison_pairs`), complete owner-native episode rows and normalized feature
and composite scores in the inclusive range `[0, 1]`. Rows marked with a
fallback, degraded, unavailable, failed, or equivalent non-admissible
execution status are rejected before canonical provenance is advertised.
Each declared `composite_score` is recomputed from the owner-native feature
values and report weights using the positive-weighted normalized formula; the
declared value must match within an absolute tolerance of `1e-9` (no relative
tolerance), or the canonical score input remains unavailable.
Comparison-pair endpoints must be episode IDs from the review bundle; a valid
top-N report may omit known bundle episodes from its scored rows. A lookalike,
foreign, duplicate, unknown-endpoint, or out-of-range report is rejected as
unavailable input and is never advertised as the canonical adapter. Missing or
unversioned scores remain unavailable; they never silently become zero.

Event rows must carry an episode ID when they carry an interval or event ID.
Duplicate `interval_id` or `event_id` values are checked within the episode
scope, so equal source IDs in different episodes remain distinct while an
ambiguous duplicate is omitted and makes the result partial. The shared
`source_intervals` field remains the renderer-neutral time list required by
`visualization-spec.v1`; the annotation's `event_intervals` list preserves the
episode, interval, event, and source-row identity alongside those times.

## Command

```bash
uv run python -m robot_sf.analysis_workbench.review_storyboard \
  --input tests/fixtures/scenario_review/review_storyboard/request.json \
  --config tests/fixtures/scenario_review/review_storyboard/config.json \
  --output output/scenario_review/srev-07-smoke
```

## Output

The CLI and Python API expose `component-result.v1` envelopes. Source bytes
are parsed as strict UTF-8 JSON where applicable (duplicate keys and
`NaN`/infinite constants are rejected), read only from bounded regular files,
and checked against declared SHA-256 values. Source and output paths are walked
relative to pinned directory descriptors with no-follow flags; an unavailable
descriptor-relative operation or detected ancestor replacement fails closed.
Bundle references are retained in result provenance with their schema, commit,
configuration, units, coordinate frame, and observed integrity status.

Three sidecars are emitted together into a new output directory:

- `storyboard-spec.json`: validated `visualization-spec.v1` with ranked
  candidates, actual source artifact IDs, score source, pin state, and clipped
  intervals.
- `override-provenance.json`: versioned applied/ignored override state and the
  declared versus observed bundle digest.
- `missing-capability-report.json`: versioned requested, available, and
  missing capability state with the same diagnostics as the result envelope.

For a `complete` result, all three sidecars appear in the result `artifacts`
array with their raw-byte digests and also in `provenance.emitted_artifacts`.
For `partial` results, the sidecars remain diagnostic output but the shared
result contract requires an empty `artifacts` array; their complete inventory
is still present in `provenance.emitted_artifacts`.

## Unavailable reasons and limits

- Missing required families or capabilities, corrupt or unbound sources,
  digest mismatches, unsafe paths, duplicate episode IDs, malformed or
  out-of-range score shapes, non-admissible canonical score rows, unknown
  comparison-pair endpoints, unscoped or duplicate event IDs, stale or
  unbound override sources, and invalid intervals never silently produce a
  complete ranking.
- An optional family that is not requested may be absent. Its capability
  remains unavailable in the capability report, and no score fallback is
  synthesized.
- Output directories must not already exist. This prevents a rerun from
  overwriting an earlier receipt; final publication reserves the directory
  without replacement semantics, links staged sidecars through pinned
  descriptors, checks ancestor identity, and removes partial output on write
  failure.
- Resource limits are explicit: 8 MiB per input file, 256 request sources,
  10,000 bundle episodes, 64 references per episode, 100,000 score rows or
  event intervals, and 10,000 override IDs per pin/exclude list. Nested bundle
  verification also caps the aggregate at 20,000 references and reads and 64
  MiB of referenced bytes, including repeated paths. Exceeding a
  required-input limit fails; exceeding an optional score or interval limit
  yields a partial diagnostic result.
- `episode-json` references are strictly parsed and, when present, their
  `artifact_id` and `episode_id` must match the bundle binding. A malformed or
  mismatched source fails closed.
- Invalid request or config JSON supplied before API validation still produces
  a schema-valid failed result envelope on stdout and exits with status 1.
- Evidence boundary: ranking reflects provided diagnostic scores and explicit
  overrides, not scientific merit. No benchmark, safety, or paper-facing claim
  follows from a storyboard or its fixtures.

## Discoverability boundary

The central scenario-review index and component registry are owned by SREV-29
and its approved owner path. This SREV-07 leaf intentionally does not edit
`docs/scenario_review/README.md` or registry files. A bounded follow-up remains:
the SREV-29 owner should add this page to the central index when an approved
owner packet authorizes that cross-leaf change.

The focused-test fast-lane registration is likewise an owner follow-up outside
this leaf's allowlist (`tests/conftest.py`). SREV-07 does not edit that central
file; until the fast-lane owner applies the registration, routing remains a
documented blocker even when the focused tests pass.
