# Issue 513 High-Density Perf Gate Calibration

## Goal

Decide whether `classic_cross_trap_high` in the overall performance trend matrix should become a
blocking regression gate or remain advisory.

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/513>

## Evidence

Local trend run on 2026-05-02:

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python -m robot_sf.benchmark.perf_trend \
  --matrix configs/benchmarks/perf_trend_matrix_classic_v1.yaml \
  --history-glob 'output/benchmarks/perf/trend/history/*.json' \
  --output-json output/benchmarks/perf/trend/issue513_latest.json \
  --output-markdown output/benchmarks/perf/trend/issue513_latest.md
```

The configured history glob found no local reports, so the run reported:

- history comparison: `no-history`
- diagnostics: `No history reports found.`

Current scenario medians from the same run:

| Scenario | Gate | Status | cold.env_create_sec | cold.first_step_sec | cold.steps_per_sec | warm.steps_per_sec |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `classic_cross_trap_low` | blocking | `warn` | 5.460 | 5.454 | 17.419 | 1589.536 |
| `classic_cross_trap_medium` | blocking | `warn` | 5.514 | 5.472 | 17.310 | 1300.508 |
| `classic_cross_trap_high` | advisory | `warn` | 5.465 | 5.478 | 17.240 | 1047.435 |

All three scenario warnings were startup-only (`cold.env_create_sec` and `cold.first_step_sec`).
No steady-state throughput gate was violated.

## Decision

Keep `classic_cross_trap_high` advisory:

- `require_baseline: false`
- `enforce_regression_gate: false`

The issue asked for a stable-history decision, and the available local evidence does not contain a
nightly history window. Promoting high-density to blocking from one local run would overfit to a
single machine and contradict the issue's noise-versus-signal requirement.

The benchmark config now carries an inline issue-513 rationale, `docs/performance_notes.md`
documents the policy, and `tests/benchmark/test_perf_trend.py` asserts that the high-density entry
remains advisory while low/medium stay blocking.

## Follow-Up Boundary

Revisit the gate only when several schema-compatible nightly reports are available under the trend
history path or a durable artifact bundle. A future promotion PR should quote the history window,
median deltas, and whether the observed degradation is startup-only or steady-state.
