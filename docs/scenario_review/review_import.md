# Review import (SREV-02)

Command-line glossary: SREV is the scenario-review portfolio; a review bundle
(`review-bundle.v1`) is an index of episode, trace, media, and diagnostic
references; a component request (`component-request.v1`) is the fixture envelope
that invokes one workbench component; shared contract shapes live in
`robot_sf/analysis_workbench/review_contracts.py` (SREV-01).

## What it does

`python -m robot_sf.analysis_workbench.review_import` imports existing benchmark
artifacts into a `review-bundle.v1` index plus a missing-capability report,
without rewriting any source payload (sources are read, hashed, and indexed by
reference only). Five source families are supported: `episode-jsonl`,
`analysis-trace`, `simulation-trace`, `trace-dossier`, and `video-metadata`
(video-only sources import without pretending telemetry exists).
`presentation-manifest` is an accepted optional stream: it is skipped with a
report entry unless the request lists it as required.

## Command

```bash
uv run python -m robot_sf.analysis_workbench.review_import \
  --input tests/fixtures/scenario_review/review_import/request.json \
  --config tests/fixtures/scenario_review/review_import/config.json \
  --output output/scenario_review/srev-02-smoke
```

`--input` is a `component-request.v1` envelope; `--config` merges extra config
(notably `source_metadata`, which declares expected SHA-256 digests, source
commits, units, and schemas per artifact id); `--output` must not exist
(collision fails closed). Exit code is 0 only when the result status is
`complete`. Source URIs resolve under the current directory (or `--base`).

## Output

- `review-bundle.json`: validated `review-bundle.v1` index (episode ids, source
  ids/hashes/units/frames, evidence status). Deterministic: repeated runs produce
  identical logical digests.
- `missing-capability-report.json`: skipped optional streams and per-source
  diagnostics.
- The printed result envelope carries `complete`, `partial`, `unavailable`, or
  `failed` with stable reason codes (`source_unreadable`, `stale_digest`,
  `duplicate_episode_id`, `unknown_source_format`, `missing_required_capabilities`,
  `incompatible_component_version`, `output_collision`, ...). Only `complete`
  results list envelope artifacts; index files stay on disk either way.

## Unavailable reasons and limits

- Missing files, corrupt lines, stale digests, and duplicate episode ids make the
  result `partial`, never silently complete.
- Unknown component ids or required capabilities yield `unavailable`; version,
  collision, and validation failures yield `failed`.
- Undeclared source commits become an explicit zero placeholder plus an
  informational diagnostic (never a fabricated provenance claim).
- Evidence boundary: indexing proves the bundle mirrors its sources, not that the
  sources are scientifically admissible; native/adapter/fallback/degraded mode and
  evidence admission stay with the source artifacts. No benchmark, safety, or
  paper-facing claim follows from an import.
