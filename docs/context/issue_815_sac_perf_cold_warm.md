# Issue 815 SAC Cold/Warm Performance Profile

## Goal

Profile the remaining simulator/environment startup cost versus steady-state stepping cost after the
SAC cache fix, using the repository cold/warm performance harness.

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/815>

## Measurement Scope

The run used the benchmark harness in `robot_sf/benchmark/perf_cold_warm.py` against
`configs/scenarios/archetypes/classic_cross_trap.yaml` scenario `classic_cross_trap_low`.

The harness separates:

- cold samples: fresh subprocess execution with one measured environment lifecycle,
- warm samples: repeated in-process measurements after one warmup measurement,
- `env_create_sec`: `make_robot_env(...)` construction time,
- `first_step_sec`: first `env.step(...)` after `env.reset(...)`,
- `episode_sec` and `steps_per_sec`: reset plus the configured episode steps.

The baseline file was `configs/benchmarks/perf_baseline_classic_cold_warm_v1.json`.

## Commands

Canonical issue command, with issue-specific output paths:

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python -m robot_sf.benchmark.perf_cold_warm \
  --scenario-config configs/scenarios/archetypes/classic_cross_trap.yaml \
  --scenario-name classic_cross_trap_low \
  --episode-steps 64 \
  --cold-runs 1 \
  --warm-runs 2 \
  --baseline configs/benchmarks/perf_baseline_classic_cold_warm_v1.json \
  --output-json output/benchmarks/perf/cold_warm_issue815_local.json \
  --output-markdown output/benchmarks/perf/cold_warm_issue815_local.md
```

Repeat run with a small additional sample count:

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python -m robot_sf.benchmark.perf_cold_warm \
  --scenario-config configs/scenarios/archetypes/classic_cross_trap.yaml \
  --scenario-name classic_cross_trap_low \
  --episode-steps 64 \
  --cold-runs 2 \
  --warm-runs 3 \
  --baseline configs/benchmarks/perf_baseline_classic_cold_warm_v1.json \
  --output-json output/benchmarks/perf/cold_warm_issue815_repeat.json \
  --output-markdown output/benchmarks/perf/cold_warm_issue815_repeat.md
```

`output/benchmarks/perf/` is worktree-local and ignored. The command and summary here are the
durable record; regenerate the output files when exact local artifacts are needed.

## Evidence

The canonical one-cold/two-warm run reported `WARN` with `failure_class=startup_only`:

| Phase | env_create_sec | first_step_sec | episode_sec | steps_per_sec |
| --- | ---: | ---: | ---: | ---: |
| cold | 5.491 | 5.436 | 5.474 | 11.692 |
| warm | 0.008 | 0.307 | 0.346 | 812.291 |

The repeat two-cold/three-warm run reported the same classification:

| Phase | env_create_sec | first_step_sec | episode_sec | steps_per_sec |
| --- | ---: | ---: | ---: | ---: |
| cold | 5.485 | 5.172 | 5.211 | 12.328 |
| warm | 0.008 | 0.001 | 0.042 | 1517.181 |

Against the stored baseline, the repeat run flagged:

| Phase | Metric | Baseline | Current | Delta | Classification |
| --- | --- | ---: | ---: | ---: | --- |
| cold | env_create_sec | 3.000 | 5.485 | +2.485 | regression |
| cold | first_step_sec | 2.000 | 5.172 | +3.172 | regression |
| cold | episode_sec | 8.000 | 5.211 | -2.789 | not regressed |
| warm | env_create_sec | 2.000 | 0.008 | -1.992 | not regressed |
| warm | first_step_sec | 0.350 | 0.001 | -0.349 | not regressed |
| warm | episode_sec | 6.000 | 0.042 | -5.958 | not regressed |

## Conclusion

The remaining measurable cost is cold startup and lazy first-step initialization, not warm
steady-state stepping. The profiler's own diagnostic is:
`Regression localized to startup overhead (env creation / first step).`

This does not justify an environment reuse or pooling change from this evidence alone. Reuse could
change reset isolation and benchmark semantics, while the warm measurements show the post-warmup
path is already far below the baseline. Any future optimization should first split cold startup
more finely, for example import time, scenario/config loading, map parsing, environment
construction, reset, and first-step backend initialization.

## Follow-Up Boundary

No code optimization is retained for issue #815. The useful handoff is the classification: target
cold/lazy initialization if SAC startup remains a practical bottleneck, and avoid spending tuning
time on steady-state simulator stepping for this scenario until broader evidence contradicts this
run.
