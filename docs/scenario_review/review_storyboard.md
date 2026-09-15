# Review storyboard (SREV-07)

Command-line glossary: SREV is the scenario-review portfolio; a storyboard
spec (`visualization-spec.v1`) is the ranked-candidate document with source
intervals; a component request (`component-request.v1`) is the fixture envelope
that invokes one workbench component; shared contract shapes live in
`robot_sf/analysis_workbench/review_contracts.py` (SREV-01).

## What it does

`python -m robot_sf.analysis_workbench.review_storyboard` ranks bundle episodes
into storyboard candidates with deterministic tie-breaking
(score-descending, then episode-id-ascending) and emits a validated
`visualization-spec.v1` document plus override provenance. User pins/excludes
apply deterministically every regeneration; a declared override source digest
that no longer matches disables overrides with an explicit stale diagnostic
instead of applying them blindly. Event intervals clip into the configured
source duration with identity preserved.

## Command

```bash
uv run python -m robot_sf.analysis_workbench.review_storyboard \
  --input tests/fixtures/scenario_review/review_storyboard/request.json \
  --config tests/fixtures/scenario_review/review_storyboard/config.json \
  --output output/scenario_review/srev-07-smoke
```

## Output

- `storyboard-spec.json`: validated spec with ranked candidates (episode, score,
  score source, pinned flag), tie-break method, and clipped source intervals.
- `override-provenance.json`: applied pins/excludes, freshness flag, diagnostics.
- The printed result envelope carries `complete`, `partial`, `unavailable`, or
  `failed` with stable reason codes. Only `complete` results list envelope
  artifacts; spec files stay on disk either way.

## Unavailable reasons and limits

- Missing families, corrupt sources, unknown override ids, stale override
  sources, and out-of-range intervals make the result `partial` or `failed`,
  never silently complete. Intended exclusions are provenance-recorded, not errors.
- Evidence boundary: ranking reflects provided scores and explicit overrides,
  not scientific merit. No benchmark, safety, or paper-facing claim follows from
  a storyboard.
