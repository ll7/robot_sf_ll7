# Review context (SREV-06)

Command-line glossary: SREV is the scenario-review portfolio; a cohort context
report (`review-context.v1`) carries denominators, outcome frequencies, metric
positions, and selection coverage; a component request (`component-request.v1`)
is the fixture envelope that invokes one workbench component; shared contract
shapes live in `robot_sf/analysis_workbench/review_contracts.py` (SREV-01).

## What it does

`python -m robot_sf.analysis_workbench.review_context` builds a cohort context
report from campaign-result inputs. Repeated excerpts never inflate unique
counts; cohort, config, and seed grain are recorded explicitly; missing metrics
retain denominator and missing count; tie percentiles use documented linear
interpolation; selection coverage reports known/unknown ids; an unresolvable
campaign reference yields unavailable context instead of a population inference.

## Command

```bash
uv run python -m robot_sf.analysis_workbench.review_context \
  --input tests/fixtures/scenario_review/review_context/request.json \
  --config tests/fixtures/scenario_review/review_context/config.json \
  --output output/scenario_review/srev-06-smoke
```

## Output

- `context-report.json`: grain, denominator, outcome frequencies, per-metric
  summaries (count/missing/min/max/mean/p25/p50/p75 plus the percentile method),
  selection coverage, and campaign availability.
- `context-report.html`: standalone table of outcomes and metric positions.
- The printed result envelope carries `complete`, `partial`, `unavailable`, or
  `failed` with stable reason codes. Only `complete` results list envelope
  artifacts; report files stay on disk either way.

## Unavailable reasons and limits

- Corrupt sources, unresolved campaign references, unknown selections, and
  missing families make the result `partial` or `failed`, never silently complete.
- Evidence boundary: the report describes the provided cohort exactly; it does
  not establish population claims, causal mechanisms, or benchmark readiness. No
  safety or paper-facing claim follows from a context report.
