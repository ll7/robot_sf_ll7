# Issue #3544 Signalized Failure-Pack Real-Run Negative Control

Evidence status: analysis-only real-run negative control.

This directory records a local simulator-backed signalized-crossing run and the issue #2754
failure-pack builder result. The run used real episode metric rows plus schema-valid trace exports
generated from embedded simulation-step traces, but the builder found no failure-pack cases under
its current failure predicate.

## Result

- Runtime rows: 4.
- Planner-observable signal denominator rows: 2.
- Fail-closed excluded rows: 2.
- Failure-pack cases: 0.
- Pack status: `insufficiently_adversarial`.
- Figure eligibility: false.

This is not traffic-light compliance evidence, not a dissertation figure artifact, and not a
planner-performance comparison. It proves the current signalized runtime smoke can produce real
denominator rows, but this specific run did not produce a failure-pack case.

## Commands

Preflight dependency note:

```bash
uv sync --extra training
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/signalized_runtime_smoke_issue_2799.yaml \
  --output-root output/benchmarks/issue_3544_signalized_real_pack_probe \
  --label issue_3544_preflight \
  --skip-publication-bundle \
  --mode preflight
```

Durable local signalized run:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/signalized_runtime_smoke_issue_2799.yaml \
  --output-root output/benchmarks/issue_3544_signalized_real_pack \
  --label issue_3544_real \
  --skip-publication-bundle
```

Trace-recording run used for the failure-pack builder:

```bash
uv run python - <<'PY'
from robot_sf.benchmark.cli import cli_main

raise SystemExit(
    cli_main(
        [
            "run",
            "--matrix",
            "configs/scenarios/single/issue_2799_signalized_runtime_smoke.yaml",
            "--out",
            "output/benchmarks/issue_3544_signalized_trace_run/episodes.jsonl",
            "--horizon",
            "80",
            "--dt",
            "0.1",
            "--record-forces",
            "--record-simulation-step-trace",
            "--no-video",
            "--algo",
            "goal",
            "--benchmark-profile",
            "baseline-safe",
            "--workers",
            "1",
            "--no-resume",
            "--structured-output",
            "json",
            "--external-log-noise",
            "suppress",
        ]
    )
)
PY
```

The trace exports were built from `algorithm_metadata.simulation_step_trace.steps`, validated as
`simulation_trace_export.v1`, and kept under the ignored local artifact root:

```text
output/benchmarks/issue_3544_signalized_trace_run/traces/
```

Failure-pack builder:

```bash
uv run python scripts/analysis/build_signalized_crossing_failure_pack_issue_2754.py \
  --traces output/benchmarks/issue_3544_signalized_trace_run/traces/*.trace.json \
  --episodes-jsonl output/benchmarks/issue_3544_signalized_trace_run/episodes.jsonl \
  --trace-source-kind live_execution \
  --metric-source-kind live_execution \
  --execution-performed \
  --evidence-tier analysis_only \
  --execution-mode native \
  --artifact-status current \
  --claim-matrix-status claimable \
  --output-json docs/context/evidence/issue_3544_signalized_failure_pack_real/summary.json
```

## Tracked Files

- `summary.json`: issue #2754 failure-pack builder output. It is a negative-control pack with zero
  cases.
- `runtime_metrics/summary.json`: compact signalized runtime row summary.
- `runtime_metrics/report.md`: human-readable runtime row report.
- `runtime_metrics/README.md`: reproduction note from the runtime metrics report generator.

## Local Artifact Checksums

Raw run artifacts remain under ignored `output/` and are not mirrored wholesale into git.

```text
f6b0a4fb669ba4f1d197e991e734efe7d57d40d15f530426868fe737057e6422  output/benchmarks/issue_3544_signalized_trace_run/episodes.jsonl
bb751dc888dfc9dd8a848598b28b7e9b5a069734c3659e937eb781d2d5acd00d  output/benchmarks/issue_3544_signalized_trace_run/traces/issue_2799_red_required_stop_observable--2799--60c725f2f37c0307.trace.json
90f60a0a92a855861c788fe2c978bc3eba2e1ab8d2ca4d7c23292d15c2f1622f  output/benchmarks/issue_3544_signalized_trace_run/traces/issue_2799_green_proceed_observable--2800--80d40eb20c230d85.trace.json
0841a05ad2f40a0bb332a61720ad0ca1e2057115359b85b2b79d28582b5d9d90  output/benchmarks/issue_3544_signalized_trace_run/traces/issue_2799_unavailable_no_claim--2801--ce49d0bf33f66d62.trace.json
21e69720bd3c24bff15fdf76b1f3f2f21436960211e7bec7adbd2e359d3e333c  output/benchmarks/issue_3544_signalized_trace_run/traces/issue_2799_proxy_only_denominator_excluded--2802--b57258021f1094d1.trace.json
```

## Follow-Up

A positive failure-pack case needs either a more adversarial signalized scenario or an explicit
issue #2754 contract update that decides whether signal-specific metrics such as
`signal_red_phase_violations` should be failure-pack predicates. Until then, this directory should
remain negative-control evidence.
