# Review context (SREV-06)

SREV is the scenario-review portfolio. SREV-06 produces a diagnostic-only
cohort context report from one explicitly selected, recorded campaign-result
source. It does not execute a simulator or establish population, benchmark,
causal, safety, or paper-facing claims.

## Contract and command

The component consumes the shared `component-request.v1` request and emits the
shared `component-result.v1` envelope. Its descriptor is available from
`robot_sf.analysis_workbench.review_context.descriptor()` and its leaf-owned
machine-readable payload registry from `output_schemas()`. The registry covers
`review-context.v1` and `missing-capability-report.v1`; the shared envelope
schemas remain owned by SREV-01.

```bash
uv run python -m robot_sf.analysis_workbench.review_context \
  --input tests/fixtures/scenario_review/review_context/request.json \
  --config tests/fixtures/scenario_review/review_context/config.json \
  --output output/scenario_review/srev-06-smoke
```

Every source reference must declare its expected family schema, SHA-256,
40-character source commit, and non-empty configuration identity. The loader
checks the exact bytes, rejects absolute/traversal/symlink paths, and requires
the source payload's schema and execution status. Multiple campaign or
selection candidates require an explicit `canonical_*_artifact_id` config key;
they are never silently merged.

## Outputs

- `context-report.json` (`review-context.v1`) records planner/scenario/config/
  seed grain, the denominator of admitted rows, outcome frequencies, missing-aware
  metric summaries, deterministic linear-interpolation percentiles, selection
  coverage, the selected campaign source, and excluded row statuses.
- `context-report.html` is a deterministic standalone rendering of the same
  diagnostic context.
- `missing-capability-report.json` (`missing-capability-report.v1`) records
  optional streams that were not supplied and structured diagnostics.
- A `complete` result lists all three files in its component-result envelope.
  Partial, unavailable, and failed results do not list usable artifacts;
  partial reports may remain on disk as diagnostic output.

Fallback, degraded, unavailable, and failed rows are excluded from the
denominator and are recorded in `exclusions`. A requested capability without an
actual valid source returns `unavailable`; malformed or conflicting source
content returns `failed`. An absent or unresolved `campaign_id` returns
`unavailable` without writing a cohort report. Unknown selections and invalid
metric values prevent a complete result and remain visible as diagnostics.

The checked-in fixture is synthetic and diagnostic-only. Generated output is
temporary under `output/` and is not a durable evidence dependency.
