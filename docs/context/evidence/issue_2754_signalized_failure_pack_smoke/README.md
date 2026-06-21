# Issue #2754 Signalized Crossing Failure-Pack Smoke Evidence

Evidence status: smoke fixture evidence only.

This directory records a first use of
`scripts/analysis/build_signalized_crossing_failure_pack_issue_2754.py` on a tracked
signalized trace fixture plus a compact synthetic metric row. It validates the pack extraction
contract for issue #2754, including trace row range, active signal phase, stop-line geometry,
robot/pedestrian state, metric row, denominator status, artifact status, and allowed claim wording.

It is not benchmark evidence, dissertation figure evidence, or traffic-light compliance evidence.
The metric row in `episodes.jsonl` is intentionally small and synthetic; it exists to exercise the
pack builder against a durable trace fixture without copying raw traces into this directory.

## Inputs

- Trace fixture:
  `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2868_signalized_crossing_fixture_0000.json`
- Metric row:
  `docs/context/evidence/issue_2754_signalized_failure_pack_smoke/episodes.jsonl`

## Reproduction

```bash
uv run python scripts/analysis/build_signalized_crossing_failure_pack_issue_2754.py \
  --traces tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2868_signalized_crossing_fixture_0000.json \
  --episodes-jsonl docs/context/evidence/issue_2754_signalized_failure_pack_smoke/episodes.jsonl \
  --artifact-status current \
  --allowed-claim-wording "Smoke fixture evidence only: this pack validates signalized failure-pack extraction fields against a tracked trace fixture and synthetic metric row; it is not benchmark evidence, dissertation figure evidence, or traffic-light compliance evidence." \
  --output-json docs/context/evidence/issue_2754_signalized_failure_pack_smoke/summary.json
```

## Output

- `summary.json`: compact `signalized_crossing_failure_pack.v1` smoke output with one
  fixture-backed case.
