# Review events (SREV-05)

Command-line glossary: SREV is the scenario-review portfolio; an event index
(`event-index.v1`) is the ordered interval map with precursor/recovery links;
a component request (`component-request.v1`) is the fixture envelope that
invokes one workbench component; shared contract shapes live in
`robot_sf/analysis_workbench/review_contracts.py` (SREV-01).

## What it does

`python -m robot_sf.analysis_workbench.review_events` indexes event and phase
intervals with explicit precursor/recovery links. Overlap and nesting retain
identities (nothing is merged); intervals clip into the configured `[t0_s,
terminal_s]` window with an explicit note; missing or dangling links resolve to
unavailable reasons (never invented); metric values and failure categories pass
through unchanged.

## Command

```bash
uv run python -m robot_sf.analysis_workbench.review_events \
  --input tests/fixtures/scenario_review/review_events/request.json \
  --config tests/fixtures/scenario_review/review_events/config.json \
  --output output/scenario_review/srev-05-smoke
```

The checked-in fixture contains one linkless phase, so its expected verdict is
`partial` with the `links_unavailable` code — asserting that proves missing
links are preserved as unavailable rather than invented.

## Output

- `event-index.json`: intervals ordered by start/end with ids, kinds, actors,
  categories, metric values, declared links, and per-interval link availability.
- `missing-capability-report.json`: schema `missing-capability-report.v1`, with
  `missing_capabilities`, `skipped_optional_streams`, and sorted diagnostics
  for optional streams and non-complete indexing conditions.
- The printed result envelope is the shared `component-result.v1` contract and
  carries `complete`, `partial`, `unavailable`, or `failed` with stable reason
  codes. Only `complete` results list envelope artifacts; index files stay on
  disk for partial results when indexing reached the write step.

## Versioned source and output contracts

Each required source reference declares its family schema (`event-list.v1` or
`phase-list.v1`) and the SHA-256 digest of the exact source bytes. The loader
checks both declarations before parsing and records the declared and observed
digests plus available source commit, configuration identity, units, and
coordinate-frame metadata in result provenance. A source that cannot be
contained below the request base, does not match its digest/schema, or is not a
JSON object with a versioned `intervals` list is not admitted.

`event-index.json` has schema `event-index.v1` and contains `t0_s`,
`terminal_s`, and `intervals`; each interval contains `interval_id`, `kind`,
`start_s`, `end_s`, `actor_ids`, `category`, `metric_value`, declared link
ids, and `links_available`/`links_reason`. The leaf documents these output
shapes here while the shared SREV-01 owner remains authoritative for the
`component-request.v1` and `component-result.v1` schemas.

## Unavailable reasons and limits

- Malformed rows, out-of-range intervals, duplicates, corrupt sources, dangling
  links, and missing links make the result `partial`, never silently complete.
- Evidence boundary: the index proves explicit correspondence between declared
  intervals and links, not that the intervals are scientifically admissible. No
  benchmark, safety, or paper-facing claim follows from an index.
