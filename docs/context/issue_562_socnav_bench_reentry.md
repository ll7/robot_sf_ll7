# Issue #562 SocNavBench Re-Entry Gate

Date: 2026-05-02
Related issue:
- <https://github.com/ll7/robot_sf_ll7/issues/562>

## Goal

Make `socnav_bench` re-entry criteria explicit before spending more benchmark time on it. The
planner remains excluded from the canonical paper-facing matrix unless it can run without fallback,
show competitive outcome quality, and stop being a runtime outlier.

## Current Status

`socnav_bench` is still a dependency-sensitive external adapter:

- `docs/benchmark_planner_family_coverage.md` classifies the SocNavBench adapter family as
  `conceptually adjacent only`.
- `configs/benchmarks/paper_experiment_matrix_v1.yaml` excludes `socnav_bench`.
- `configs/benchmarks/paper_experiment_matrix_all_planners_v1.yaml` keeps `socnav_bench` only in
  the all-planners profile and sets `socnav_missing_prereq_policy: fail-fast`.
- `robot_sf/benchmark/map_runner.py` wires `socnav_bench` through
  `SocNavBenchSamplingAdapter` with `allow_fallback=False` unless an algo config explicitly opts
  into fallback.

Fallback execution is not acceptable evidence for re-entry.

## Local Prerequisite Check

The local asset check was run with:

```bash
uv run python scripts/tools/prepare_socnav_assets.py \
  --report-json output/tmp/issue562_socnav_asset_report.json
```

Result: required schematic assets are missing on this machine:

- `wayptnav_data`
- `sd3dis/stanford_building_parser_dataset`
- `sd3dis/stanford_building_parser_dataset/traversibles`

The generated JSON report is intentionally under ignored `output/` because these assets are local
and license-sensitive.

## Local Probe Result

The focused config shape was validated with:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/socnav_bench_reentry_probe.yaml \
  --mode preflight \
  --label issue562_preflight
```

This wrote:

- `output/benchmarks/camera_ready/socnav_bench_reentry_probe_issue562_preflight_20260502_041059`

That preflight validates matrix shape, scenario resolution, and seed expansion; it does not by
itself prove SocNavBench asset availability.

The fail-fast execution probe was then run with:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/socnav_bench_reentry_probe.yaml \
  --label issue562_failfast_probe \
  --log-level INFO
```

This wrote:

- `output/benchmarks/camera_ready/socnav_bench_reentry_probe_issue562_failfast_probe_20260502_041126`

Observed outcome:

- `goal`: `status=ok`, `episodes=3`, `runtime_sec=10.8862`
- `socnav_bench`: `status=failed`, `episodes=0`, `runtime_sec=0.0224`
- campaign: `benchmark_success=false`
- failure: missing SocNavBench control-pipeline assets, starting with
  `third_party/socnavbench/wayptnav_data`

This is the intended fail-closed behavior. It is not a planner-quality or runtime-overhead profile
because `socnav_bench` never entered episode execution.

## Focused Probe Config

New focused config:

```bash
configs/benchmarks/socnav_bench_reentry_probe.yaml
```

The probe deliberately stays small:

- one scenario: `configs/scenarios/single/francis2023_blind_corner.yaml`
- fixed seeds: `111, 112, 113`
- horizon: `30`
- workers: `1`
- planners: `goal`, `socnav_bench`
- `socnav_bench` prerequisite policy: `fail-fast`

The `goal` row provides a cheap sanity baseline for the exact same scenario/seed slice. The config
does not include PPO or other expensive planners because the first re-entry question is whether
`socnav_bench` can run dependency-backed and produce non-fallback metrics.

## Re-Entry Criteria

`socnav_bench` should not re-enter a paper-facing matrix until all of these are true:

1. Asset validation passes for schematic mode using `scripts/tools/prepare_socnav_assets.py`.
2. A focused probe run on `socnav_bench_reentry_probe.yaml` completes all `3` episodes.
3. The focused probe records `success_mean > 0.0`; zero-success behavior is a keep-out result.
4. Runtime is no worse than `3x` the `goal` baseline on the same probe slice.
5. A broader follow-up run on a paper-matrix-compatible subset confirms the same status without
   fallback before any canonical paper matrix change.

These thresholds are intentionally conservative and local to re-entry triage. Passing them would
justify broader evaluation, not automatic promotion.

## Recommendation

Recommendation: `keep out of paper matrix; use focused fail-fast probe before re-entry work`

The current blocker is prerequisite/data availability on this machine, not a code-level performance
profile. Once assets are available, the next measured step should run the focused probe and compare
`goal` vs `socnav_bench` runtime and success on the same deterministic seeds.
