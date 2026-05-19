# Issue #1319 Task Bundles

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1319>

## Decision

Task bundles are a lightweight, config-first grouping layer over existing scenario manifests. Version
`robot_sf.task_bundle.v1` lives under `configs/bundles/` and resolves via `bundle:<name>` references
through the existing scenario loader.

The first bundle, `bundle:sanity-smoke-v1`, expands `configs/scenarios/sanity_v1.yaml` without
copying the scenario list into a second matrix.

## Boundary

Bundles may reference durable scenario YAML files only. The loader rejects scenario file inputs under
the repository `output/` tree so generated worktree-local files do not become implicit dependencies.

Existing scenario loading remains unchanged unless a path is a bundle YAML file or the caller uses a
`bundle:<name>` reference. Legacy multi-document matrix streams still load through
`robot_sf.benchmark.runner.load_scenario_matrix`.

## Validation

Targeted checks run on 2026-05-18:

```bash
uv run ruff check robot_sf/training/task_bundles.py robot_sf/training/scenario_loader.py robot_sf/benchmark/runner.py robot_sf/benchmark/cli.py tests/training/test_task_bundles.py tests/benchmark/test_runner_scenario_matrix_manifest.py tests/test_scenario_schema.py
uv run ruff format --check robot_sf/training/task_bundles.py robot_sf/training/scenario_loader.py robot_sf/benchmark/runner.py robot_sf/benchmark/cli.py tests/training/test_task_bundles.py tests/benchmark/test_runner_scenario_matrix_manifest.py tests/test_scenario_schema.py
uv run pytest tests/training/test_task_bundles.py tests/benchmark/test_runner_scenario_matrix_manifest.py tests/test_scenario_schema.py -q
uv run robot_sf_bench list-scenarios --matrix bundle:sanity-smoke-v1
uv run robot_sf_bench validate-config --matrix bundle:sanity-smoke-v1
uv run robot_sf_bench preview-scenarios --matrix bundle:sanity-smoke-v1
```

The CLI list smoke expanded these four scenarios in order:

```text
planner_sanity_simple
empty_map_8_directions_east
goal_behind_robot
single_ped_crossing_orthogonal
```
