# Issue #3169 Map-Runner Split Completion

Related issue: [#3169](https://github.com/ll7/robot_sf_ll7/issues/3169)
Related PR: [#3180](https://github.com/ll7/robot_sf_ll7/pull/3180)

## Current Decision

`robot_sf/benchmark/map_runner.py` should remain the compatibility and public batch-entry shell for
`run_map_batch`, planner policy construction, and older private test imports. Episode execution and
batch execution should live in owned helper modules:

- `robot_sf/benchmark/map_runner_episode.py` owns single episode rollout, metric finalization,
  trace payload assembly, and episode record construction.
- `robot_sf/benchmark/map_runner_batch_plan.py` owns batch kinematics resolution, seed-job
  expansion, and worker fixed-parameter payload assembly.
- `robot_sf/benchmark/map_runner_batch_runner.py` owns serial/parallel job execution and validated
  JSONL append ordering.
- `robot_sf/benchmark/map_runner_batch_summary.py` owns completed-batch summary and worker metadata
  accumulation.

This is a refactor boundary only. It does not change benchmark semantics, planner outputs, metric
interpretation, fallback policy, or evidence strength.

## Compatibility Alias Policy

Private `_...` imports from `robot_sf.benchmark.map_runner` still exist for older tests and scripts
that predate the split. They are compatibility aliases or thin wrappers, not the preferred
ownership location for new code.

New tests and scripts should import extracted helpers from their owning modules when practical. If
an older compatibility import is touched, either move that import to the owning helper module or
leave a focused compatibility assertion explaining why the alias remains.

The `_run_map_episode` symbol is intentionally kept in `map_runner.py` as a thin compatibility
wrapper around `map_runner_episode.run_map_episode` because worker tests and downstream diagnostic
tests monkeypatch that entry point.

## Guardrail

`tests/benchmark/test_map_runner_utils.py::test_map_runner_execution_boundaries_stay_extracted`
guards the current boundary by asserting that:

- map-runner episode execution delegates to `map_runner_episode.run_map_episode`;
- batch execution delegates to `map_runner_batch_runner.execute_map_jobs`;
- planning and summary helpers remain owned by their helper modules;
- the `_run_map_episode` compatibility wrapper does not regain the rollout loop.

## Validation Boundary

Minimum validation for future changes to this split:

- `uv run pytest tests/benchmark/test_map_runner_utils.py -q`
- focused tests for any touched planner/trace/metric helper module;
- Ruff check/format for touched benchmark and test files;
- `git diff --check`.

Run the full PR readiness gate when a future change alters benchmark behavior, planner contracts,
metrics, schema, or public command behavior.
