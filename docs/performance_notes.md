# Performance Notes - Social Navigation Benchmark

**Purpose**: Document performance baselines, benchmarks, and optimization targets for the Social Navigation Benchmark platform.

## Performance Baselines

### Hot-Path Optimization Wave Synthesis (2026-06-04)

Issue #2214 compared the pre-wave baseline
`5eead086fceac0bbdd00bb10ec133a612dfc5b25` with current main
`5bd87d58d4869e8420943301e45de1a8dc6513a1` after the simulator hot-path
allocation and snapshot-reuse PRs. The local smoke surfaces did not show a broad
end-to-end speedup: reset throughput was 4.8% lower and one-candidate policy
smoke runtime was 2.0% higher on the current revision, while environment
creation was 1.8% faster but still warning-classified.

Treat this as diagnostic, startup-dominated evidence rather than a performance
claim. The stronger recent wall-clock signal remains the issue #2172 worker
scaling diagnostic, where the 18-job h80 nominal-sanity run improved by 1.793x
from one to two workers. See
`docs/context/issue_2214_hot_path_synthesis.md` for the derived evidence and
follow-up direction.

### Environment Performance (as of 2025-01-19)

**Environment Creation**:
- Target: < 2.0 seconds
- Measured: ~1.16 seconds
- Status: ✅ PASS

**Environment Reset**:
- Target: > 1 reset/second
- Measured: ~1,745 resets/second (0.6ms/reset)
- Status: ✅ PASS

### Historical Performance Targets

From `dev_guide.md` expected ranges:
- **Environment creation**: < 1 second
- **Model loading**: 1–5 seconds
- **Simulation performance**: ~22 steps/second (~45ms/step)
- **Build time**: 2–3 minutes (first time)
- **Test suite**: 2–3 minutes (≈170 tests)

## Performance Validation

### Smoke Test Suite
Location: `scripts/validation/performance_smoke_test.py`

**Validation Criteria**:
- Environment creation < 3.0s (allows headroom)
- Reset performance > 0.5 resets/sec (minimum acceptable)
- Consistent performance across multiple runs
- Advisory startup/steady attribution fields:
  - `step_loop.first_step_sec`
  - `step_loop.step_loop_sec`
  - `step_loop.steady_step_loop_sec`
  - `step_loop.steps_per_sec`
  - `step_loop.steady_steps_per_sec`
  - `step_loop.measurement_mode`
  - `step_loop.warmup_excluded`
  - `step_loop.warmup_first_step_sec`, `step_loop.warmup_step_loop_sec`, and
    `step_loop.warmup_steps_per_sec` when explicit warmup attribution is requested

Large-crowd profiling now emits an additive `step_profile` contract block:
- `step_profile.advisory` (always `true` for this profile)
- `step_profile.gating` (`"non-gating"`)
- `step_profile.scenario_id`, `step_profile.scenario_name`, `step_profile.scenario_path`
- `step_profile.density` and `step_profile.density_advisory` when present in scenario metadata
- `step_profile.step_samples`, `step_profile.first_step_sec`
- `step_profile.step_loop_sec`
- `step_profile.steady_step_loop_sec`, `step_profile.steady_steps_per_sec`
- `step_profile.steps_per_sec`
- `step_profile.measurement_mode`, `step_profile.warmup_excluded`
- `step_profile.warmup_first_step_sec`, `step_profile.warmup_step_loop_sec`, and
  `step_profile.warmup_steps_per_sec` when explicit warmup attribution is requested
- `step_profile.pedestrian_count` (when observed after reset/step). The fast path tracked in
  [#3025](https://github.com/ll7/robot_sf_ll7/issues/3025) and
  [PR #3114](https://github.com/ll7/robot_sf_ll7/pull/3114) reuses the first measured step's
  pedestrian occupancy snapshot in
  [`scripts/validation/performance_smoke_test.py`](../scripts/validation/performance_smoke_test.py),
  with contract coverage in
  [`tests/perf/test_large_crowd_step_profile_contract.py`](../tests/perf/test_large_crowd_step_profile_contract.py),
  so the profiler avoids an extra env create/reset/step pass solely for count collection.

These fields are advisory and are for diagnostic reproducibility; they are not benchmark gates.

The step-loop fields help classify local slowdowns but do not add hard smoke-test gates by
default. Use `--step-samples` to tune the advisory sample count for local diagnosis. By default,
`first_step_sec`, `step_loop_sec`, and `steps_per_sec` measure the cold first step after reset.
Use `--warmup-steps` only when you intentionally want warm-start attribution; warm-start output is
flagged with `measurement_mode` and `warmup_excluded` so it cannot be mistaken for cold-start
performance evidence.

**Usage**:
```bash
# Run performance smoke test
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/performance_smoke_test.py --step-samples 10

# Results saved to: output/benchmarks/performance_smoke_test.json

# High-density advisory snapshot for [issue #3025](https://github.com/ll7/robot_sf_ll7/issues/3025)
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/performance_smoke_test.py \
    --scenario configs/scenarios/archetypes/classic_group_crossing.yaml \
    --scenario-name classic_group_crossing_high \
    --step-samples 3 --num-resets 1 \
    --json-output output/benchmarks/perf/issue_3025_smoke.json

# Explicit warm-start attribution for startup/JIT diagnosis only; not a runtime speedup claim
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/performance_smoke_test.py \
    --scenario configs/scenarios/archetypes/classic_group_crossing.yaml \
    --scenario-name classic_group_crossing_high \
    --step-samples 3 --warmup-steps 1 --num-resets 1 \
    --json-output output/benchmarks/perf/issue_3025_warmup_attribution.json
```

### Cold/Warm Regression Suite
Location: `robot_sf/benchmark/perf_cold_warm.py`

This suite separates cold-start and steady-state runs and compares medians against
a tracked baseline snapshot (`configs/benchmarks/perf_baseline_classic_cold_warm_v1.json`).

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python -m robot_sf.benchmark.perf_cold_warm \
    --scenario-config configs/scenarios/archetypes/classic_cross_trap.yaml \
    --scenario-name classic_cross_trap_low \
    --episode-steps 64 \
    --cold-runs 1 \
    --warm-runs 2 \
    --baseline configs/benchmarks/perf_baseline_classic_cold_warm_v1.json \
    --output-json output/benchmarks/perf/cold_warm_local.json \
    --output-markdown output/benchmarks/perf/cold_warm_local.md
```

The report includes:
- `env_create_sec`
- `first_step_sec`
- `episode_sec`
- `steps_per_sec`

and classifies regressions as startup-dominated vs steady-state-dominated.

CI integration:
- PR smoke: `.github/workflows/ci.yml` (`Cold/warm perf regression smoke`)
  - advisory on PRs with a conservative profile:
    - baseline: `configs/benchmarks/perf_baseline_classic_cold_warm_v1.json`
    - thresholds: `max_slowdown_pct=0.75`, `max_throughput_drop_pct=0.60`
    - absolute deltas: `min_seconds_delta=0.20`, `min_throughput_delta=1.00`
  - enforced on `main`/`workflow_dispatch` with default profile:
    - baseline: `configs/benchmarks/perf_baseline_classic_cold_warm_v1.json`
    - thresholds: `max_slowdown_pct=0.60`, `max_throughput_drop_pct=0.50`
    - absolute deltas: `min_seconds_delta=0.15`, `min_throughput_delta=0.75`
- Nightly broader checks: `.github/workflows/perf-nightly.yml`

### Overall Trend Benchmark Suite
Location: `robot_sf/benchmark/perf_trend.py`

This suite executes a stable scenario matrix and emits a single schema-versioned report:

- matrix config: `configs/benchmarks/perf_trend_matrix_classic_v1.yaml`
- report schema: `benchmark-perf-trend-report.v1`
- fixed KPI set per phase: `env_create_sec`, `first_step_sec`, `episode_sec`, `steps_per_sec`

Local run:

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python -m robot_sf.benchmark.perf_trend \
    --matrix configs/benchmarks/perf_trend_matrix_classic_v1.yaml \
    --history-glob 'output/benchmarks/perf/trend/history/*.json' \
    --output-json output/benchmarks/perf/trend/latest.json \
    --output-markdown output/benchmarks/perf/trend/latest.md
```

Nightly behavior:
- runs the matrix in `.github/workflows/perf-nightly.yml`
- restores prior trend reports from cache when available
- compares current run against recent history medians
- stores latest report back into history cache path
- uploads all perf artifacts from `output/benchmarks/perf/`

Regression diagnostics include whether degradation is startup-dominated
(`env_create_sec`/`first_step_sec`) or steady-state-dominated (`episode_sec`/`steps_per_sec`).
If a warm `episode_sec` or `steps_per_sec` history regression is explained by same-phase
`first_step_sec` overhead, the history comparison reports a warning and lists the startup-coupled
steady metrics instead of failing the nightly gate as a steady-state regression.

High-density gate policy:
- `classic_cross_trap_low` and `classic_cross_trap_medium` are blocking in the matrix.
- `classic_cross_trap_low` uses three warm samples so one transient first-step outlier cannot
  control an even-sample median.
- `classic_cross_trap_high` remains advisory (`enforce_regression_gate: false`) until enough
  nightly history exists to distinguish stable degradation from hardware/runtime noise.
- Issue #513 local calibration on 2026-05-02 found no local history reports under
  `output/benchmarks/perf/trend/history/*.json`; the high-density run stayed startup-only `warn`
  with warm throughput above the stored medium baseline. Do not promote this scenario to blocking
  from a one-off local run.

### Baseline Management

Use the committed snapshot as the initial reference point. If hardware/runtime
changes make it stale, regenerate values from nightly artifacts and update
`configs/benchmarks/perf_baseline_classic_cold_warm_v1.json` and
`configs/benchmarks/perf_baseline_classic_cold_warm_medium_v1.json` in a dedicated PR.

### Simulation Throughput Guard (cluster-aware)
Location: `tests/perf/test_simulation_speed_perf.py`

The throughput guard supports environment-specific calibration:

- `ROBOT_SF_SIM_STEPS_SOFT` (default `2.0`)
- `ROBOT_SF_SIM_STEPS_HARD` (default `0.5`)
- `ROBOT_SF_PERF_ENFORCE=1` to turn soft/hard threshold breaches into test failures

On shared or heterogeneous cluster nodes, keep enforcement off for exploratory runs and
set tuned thresholds per hardware profile.

### Benchmark Runner Performance
Location: `robot_sf/benchmark/runner.py`

**Parallel Processing**:
- Multi-worker support for episode generation
- Process-based parallelism with resume capabilities
- Manifest-driven episode deduplication

**Expected Throughput**:
- Single worker: ~1-2 episodes/second (varies by scenario complexity)
- 4 workers: ~4-6 episodes/second (near-linear scaling)
- Resume overhead: minimal (manifest lookup < 1ms per episode)

## Performance Monitoring

### Key Metrics to Track
1. **Environment Initialization**: Creation + first reset time
2. **Episode Generation**: Episodes per second in benchmark runs
3. **Memory Usage**: Peak RSS during multi-worker runs
4. **I/O Performance**: JSONL write throughput for large episode batches

### Regression Detection
- Performance degradation > 50% should trigger investigation
- Compare against `configs/benchmarks/perf_baseline_classic_cold_warm_v1.json`
- Use `output/benchmarks/perf/*.json` and nightly artifacts for trend monitoring
- JSONL append throughput uses a lightweight synthetic guard in
  `tests/training/test_runtime_helpers.py`. Tune `ROBOT_SF_JSONL_APPEND_RECORDS` and
  `ROBOT_SF_JSONL_APPEND_SOFT_RECORDS_PER_SEC` for local diagnosis; keep
  `ROBOT_SF_PERF_ENFORCE=0` on heterogeneous machines unless a hardware-specific threshold is
  intentional.

## Optimization Notes

### Multi-Scenario Map-Cache Profile
Location: `robot_sf/benchmark/perf_scenario_cache.py`

Measures LRU cache hit/miss/eviction rates during scenario-switching training,
isolating SVG parsing overhead from per-episode env-creation cost.

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python -m robot_sf.benchmark.perf_scenario_cache \
    --scenario-config configs/scenarios/classic_interactions.yaml \
    --repetitions 3 \
    --output-json output/benchmarks/perf/scenario_cache_profile.json \
    --output-markdown output/benchmarks/perf/scenario_cache_profile.md
```

**Findings (2026-04-13, issue #806)**:

- `classic_interactions.yaml` contains 22 scenarios across **12 unique maps**.
- The previous `_load_map_definition` LRU cache had `maxsize=8`, guaranteeing
  cache eviction for at least 4 maps on every training run touching all scenarios.
- **Fix**: `maxsize` raised to **256** — covers all current maps (75 SVGs repo-wide)
  and leaves room for growth without evictions.
- The instrumentation helper `map_cache_info()` (exported from
  `robot_sf.training.scenario_loader`) lets you verify hit/miss counts at runtime.

### Known Performance Bottlenecks
1. **Cold backend/map startup**: first-process map parsing and FastPysf/JIT setup dominate
   headless cold starts. Pygame/SDL is no longer imported by
   `make_robot_env(debug=False)` after #1290; it is still initialized on explicit
   rendering, image-observation, occupancy-grid rendering, or video paths.
2. **FastPysf compilation**: JIT overhead on first use
3. **Large episode JSON**: Serialization cost grows with trajectory length
4. **File I/O**: JSONL append becomes slow with very large files
5. **Map definition cache eviction** (fixed in #806): prior `maxsize=8` caused
   repeated SVG parsing during multi-scenario SAC training runs with >8 unique maps.

### Optimization Strategies
1. **Environment Reuse**: Keep environments alive across episodes when possible
2. **Batch Processing**: Group similar scenarios to reduce setup overhead
3. **Memory Management**: Monitor RSS growth in long-running processes
4. **Disk I/O**: Use SSD storage for results directories

## Hardware Dependencies

### Development Environment Specs
- **CPU**: M-series Apple Silicon (ARM64)
- **Memory**: 16+ GB recommended for multi-worker runs
- **Storage**: SSD for the repository's `output/` artifact directory
- **Display**: Headless mode using SDL_VIDEODRIVER=dummy

### Performance Scaling
- **Single-core**: Environment creation and reset dominated by setup costs
- **Multi-core**: Near-linear scaling up to 4-8 workers depending on scenario complexity
- **Memory**: ~500MB per worker process typical

## Validation Scripts Status

### Available Validation Scripts
- ✅ `test_basic_environment.sh`: Environment creation smoke test
- ✅ `test_model_prediction.sh`: Model loading and inference test
- ✅ `performance_smoke_test.py`: Performance baseline measurement
- ⏳ `test_complete_simulation.sh`: Full episode simulation (may timeout on complex scenarios)

### Usage in CI/CD
```bash
# Run all validation scripts
./scripts/validation/test_basic_environment.sh
./scripts/validation/test_model_prediction.sh
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/performance_smoke_test.py
```

## Performance History

### 2025-01-19 - Initial Baseline
- Environment creation: 1.16s (target: < 2.0s)
- Environment reset: 1,745 resets/sec (target: > 1/sec)
- Platform: macOS ARM64, Python 3.13, headless mode
- Status: All performance targets met

---
**Monitoring Schedule**: Monthly performance validation recommended
**Next Review**: 2025-02-19
**Contact**: Reference dev team for performance regression issues
