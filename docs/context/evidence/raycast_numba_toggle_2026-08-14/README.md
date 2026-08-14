<!-- AI-GENERATED - NEEDS-REVIEW -->
# Raycast numba on/off micro-benchmark

Per-call wall time of `robot_sf.sensor.range_sensor.raycast` with numba JIT
enabled versus `NUMBA_DISABLE_JIT=1`, on a fixed synthetic LiDAR workload.

## Result

| Arm | median | IQR | min | max | n |
| --- | --- | --- | --- | --- | --- |
| numba ON | 25.79 us | 1.17 us | 24.42 us | 604.21 us | 6000 |
| numba OFF | 16.587 ms | 0.393 ms | 15.333 ms | 50.156 ms | 6000 |

Median speedup: **643.1x**
(per-repeat median ratios span 630.5x to 664.8x).

Both arms produced an identical scan output signature
(`outputs_identical_across_arms = True`), so the
comparison is between two implementations of the same computation.

## Protocol

* Workload: 272 rays x 60 obstacle segments x
  30 pedestrians, seed 101, 10.0 m max scan
  distance, 20.0 m map extent.
* 50 discarded warmup calls per arm, so JIT compilation is excluded and
  the reported numbers are steady state.
* 3 independent repeat blocks of 2000 timed calls per
  arm; the full per-call distribution is retained, not a headline mean.
* Each arm runs in its own subprocess because numba reads the JIT toggle once at import.
* Timer: `time.perf_counter_ns`.

## Identity

* Commit: `389b603658dce9beae98b63394dda218bf57d8bb`
* Hardware: Apple M4 Pro (Mac16,8),
  macOS-26.5.2-arm64-arm-64bit-Mach-O
* Environment: Python 3.13.1, numba 0.66.0,
  numpy 2.4.6
* Started / finished (UTC): 2026-08-14T15:41:29Z / 2026-08-14T15:43:14Z

## Boundary

Diagnostic hot-path timing on one machine and one synthetic workload. It is **not** a
whole-simulation speedup figure and **not** a cross-platform guarantee. Whole-simulation gain
is workload-dependent and much smaller on near-empty scenes.

## Reproduce

```bash
uv run python scripts/perf/raycast_numba_toggle_benchmark.py \
    --output-dir docs/context/evidence/raycast_numba_toggle_2026-08-14
```

## Files

* `run_metadata.json` - commit, config, environment and hardware identity.
* `raycast_numba_toggle_results.json` - full distribution summary for both arms.
* `raycast_numba_toggle_samples.csv` - every timed call (12000 rows).
* `SHA256SUMS` - integrity manifest over the files above.
