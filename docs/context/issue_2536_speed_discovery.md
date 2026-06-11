# Issue #2536 Simulator-Speed Candidate Discovery

Issue: [#2536](https://github.com/ll7/robot_sf_ll7/issues/2536)
Child issue: [#2636](https://github.com/ll7/robot_sf_ll7/issues/2636)

## Goal

Create the next bounded simulator-speed candidate after the active research queue was exhausted,
without implementing the optimization in the discovery issue. The candidate must be locally
implementable, have a clear proof path, and avoid weakening simulator correctness or benchmark
semantics.

## Evidence Checked

Searches:

```bash
gh issue list --state open \
  --search 'repo:ll7/robot_sf_ll7 is:issue is:open speed OR performance OR profiler OR bottleneck' \
  --limit 50 --json number,title,labels,url
rg -n "performance|speed|profiler|hot path|bottleneck|slow|runtime|throughput" \
  docs memory robot_sf fast-pysf scripts tests
rg -n "FastPysf compilation|JIT overhead|performance_notes|circle|raster|cell bounds|precompute" \
  docs robot_sf fast-pysf scripts tests .github
gh issue list --state open \
  --search 'repo:ll7/robot_sf_ll7 is:issue is:open (FastPysf OR JIT OR numba OR rasterize OR occupancy OR performance OR speed)' \
  --limit 60 --json number,title,labels,url
```

Key files and notes:

- `docs/performance_notes.md`: records cold backend/map startup, FastPysf/JIT first-use overhead,
  large episode JSON, and file I/O as known performance bottlenecks.
- `docs/context/simulation_speed_autoresearch_2026-04-22.md`: prior steady-state speed work moved
  `robot_sf/nav/occupancy.py::circle_collides_any_lines` into Numba and retained changes only when
  fixed-seed state hashes, rewards, reset counts, and pedestrian counts matched.
- `docs/context/issue_2214_hot_path_synthesis.md`: recent allocation/snapshot-reuse work did not
  prove broad end-to-end speedup on startup-dominated smoke; recommends representative profiles and
  semantic equivalence guards before another micro-optimization wave.
- `docs/context/issue_2172_benchmark_worker_scaling.md`,
  `docs/context/issue_2302_benchmark_worker_scaling.md`, and
  `docs/context/issue_2304_benchmark_worker_scaling_stress.md`: worker scaling remains the
  strongest recent wall-clock evidence, but current notes deliberately avoid changing global
  defaults without broader host evidence.
- `docs/dev/issues/circle-rasterization-fix/README.md`: documents the correctness fix for
  outside-grid circle overlap and explicitly leaves "precompute cell bounds for hot paths" as a
  performance opportunity.
- `robot_sf/nav/occupancy_grid_utils.py::get_affected_cells` currently loops over candidate cells
  and recomputes cell centers and cell bounds per cell before circle-rectangle intersection.
- `robot_sf/nav/occupancy_grid_rasterization.py::rasterize_circle_fast` already contains a
  vectorized bounding-slice path used for pedestrian and robot rasterization.

## Existing Open Work Classification

The live open issue search found no unclaimed ready issue that already covers a bounded local
simulator-speed implementation candidate. Nearby open items were either this discovery issue,
blocked analysis/training/data work, or running SLURM work. Worker-scaling issues are already
documented as evidence, not an immediate safe default-change task.

The delegated scout suggested FastPysf/Numba first-use mitigation. That is a real documented
bottleneck, but it is broader and riskier than the rasterization cell-bound candidate because the
proof would need separate cold-start and warm-step interpretation and may affect compiled
signatures or cache behavior. It remains a plausible later candidate after the narrower raster path
is handled.

## Recommended Child Issue

Opened [#2636](https://github.com/ll7/robot_sf_ll7/issues/2636) to optimize occupancy-grid
circle affected-cell bounds:

- Candidate surface:
  - `robot_sf/nav/occupancy_grid_utils.py::get_affected_cells`
  - `robot_sf/nav/occupancy_grid_rasterization.py::rasterize_circle`
  - `robot_sf/nav/occupancy_grid_rasterization.py::rasterize_circle_fast`
  - `tests/test_occupancy_circle_overlap.py`
  - `tests/test_occupancy_grid.py`
- Suspected bottleneck: `get_affected_cells` recomputes grid-cell world centers, per-cell min/max
  bounds, and scalar closest-point distances inside nested Python loops. This path is correctness
  oriented and still referenced by `rasterize_circle`, while the faster rasterization entry point
  shows an existing vectorized bounding-slice pattern.
- Candidate implementation boundary: add a shared helper or local vectorized/precomputed
  row/column-bound path for circle affected cells while preserving the current
  circle-rectangle-overlap semantics for circles whose centers are outside the grid.
- Proof path:
  - `uv run pytest tests/test_occupancy_circle_overlap.py tests/test_occupancy_grid.py`
  - `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy ROBOT_SF_PERF_ENFORCE=0 uv run pytest tests/perf/test_simulation_speed_perf.py -k test_simulation_step_throughput -q`
  - Include a small before/after timing harness or pytest benchmark for
    `get_affected_cells` on inside-grid and outside-grid overlap circles over a dense grid.
- Risk boundary:
  - Do not change occupancy semantics, grid indexing, obstacle/pedestrian values, or benchmark
    labels.
  - Keep outside-grid overlap tests as the primary correctness guard.
  - Treat timing as local diagnostic evidence unless repeated on a representative benchmark slice.
  - If correctness tests fail or timing is neutral/regressive, close the child as negative or
    diagnostic-only rather than forcing the optimization.

## Current Conclusion

The next safest bounded simulator-speed child is [#2636](https://github.com/ll7/robot_sf_ll7/issues/2636),
the occupancy-grid circle affected-cell bound optimization. Confidence is about 0.78: the source and
prior note identify a concrete hot path, but the actual end-to-end simulator benefit still needs
measurement because occupancy-grid rendering and observation paths are workload-dependent.
