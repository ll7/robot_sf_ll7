# SREV-08 review alignment

Align one scenario comparison with explicit compatibility checks, so
researchers can judge whether two recorded traces admit a shared
interpretation without waiting for the whole workbench. See the
[`docs/glossary.md`](../glossary.md) for project terms such as review
bundle and component result.

## Evidence boundary

Diagnostic tooling only. Alignment launches nothing — no simulation,
training, remote scheduling, or scientific publication — and admits no
scientific claim, benchmark result, or planner/simulator behavior change.
Incompatible initial states, absent shared prefixes, missing anchors, and
unequal durations stay visible in the report; per-side unavailable tails are
reported, never interpolated, and durations are never normalized.

## Interfaces

- **Input**: a `component-request.v1` request for component
  `srev08-review-alignment`. Sources cite trace geometry by identity; the
  citable format is `simulation_trace_export.v1` (validated through
  `robot_sf.analysis_workbench.simulation_trace_export`). Source URIs resolve
  relative to the request file's directory. Other formats are skipped with a
  diagnostic and never block the named comparison sides.
- **Config**: an `alignment` mapping with `left_artifact_id`,
  `right_artifact_id` (must differ), `comparison_grain`
  (`matched_planner_pair` or `matched_realization_pair`), an `anchor`
  policy (`event` with `event_id`, or `absolute-time` with `time_s`), and
  optional positive tolerance overrides (`position_tolerance_m`,
  `heading_tolerance_rad`, `shared_prefix_steps`, defaulting to the owner
  values).
- **`run(request)`**: returns a `component-result.v1` result with status
  `complete` (artifacts `alignment.json` plus `component-descriptor.json`;
  inadmissible interpretations stay visible, never silent), `partial` (extra
  sources skipped), `unavailable` (unsupported component, capability,
  grain, version, or evidence, with an actionable reason), or `failed`
  (corrupt config, missing comparison sides, or output collision; failed
  outputs never carry artifacts).
- **Output**: one `review-alignment.v1` document with the owner-computed
  compatibility record, the resolved anchor assessment, per-side durations,
  and an interpretation that is admissible only for a compatible comparison
  with a resolved anchor. `descriptor()` exposes the versioned capability
  descriptor.

Unknown required versions fail; output collisions and unsafe paths are
rejected. The alignment copies the source identities accepted by the v1
request schema; that schema carries no source-byte hash. Event anchors reuse
planner-reported event IDs from both traces, explicitly marked as reported
rather than independently detected. Repeated fixture runs compare equal
artifact bytes.

## Usage

```bash
uv run pytest tests/analysis_workbench/test_review_alignment.py -q
uv run python -m robot_sf.analysis_workbench.review_alignment \
  --input tests/fixtures/scenario_review/review_alignment/request.json \
  --config tests/fixtures/scenario_review/review_alignment/config.json \
  --output output/scenario_review/srev-08-smoke
```

`run(request)` aligns the comparison offline; `descriptor()` lists what this
component provides. Outputs are written atomically into the component-owned
output directory, which must not exist beforehand. Generated output stays
under ignored `output/` and is disposable.
