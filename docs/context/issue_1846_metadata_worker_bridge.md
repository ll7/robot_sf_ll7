# Issue #1846 Metadata Worker Bridge - 2026-05-31

Related issue:

- Issue #1846: <https://github.com/ll7/robot_sf_ll7/issues/1846>

## Outcome

Map-runner batch metadata now has an explicit worker bridge helper:

- `robot_sf/benchmark/map_runner.py::_apply_worker_metadata_bridge`

Both serial direct-call execution and parallel worker execution use this helper to fold each
episode record's `algorithm_metadata` into the batch-level `algorithm_metadata_contract`. The helper
also preserves the existing adapter-impact and kinematics-feasibility accumulators, so legacy worker
records without `algorithm_metadata` remain valid.

## Regression Guard

`tests/benchmark/test_map_runner_utils.py::test_run_map_batch_parallel_preserves_runtime_metadata_bridge`
uses the parallel map-runner path with a thread-backed executor and fake out-of-order worker
completion. It proves that worker-returned metadata survives the serialized bridge into:

- `planner_kinematics`
- `upstream_reference`
- `adapter_impact`
- `kinematics_feasibility`

No benchmark metrics, interpretation rules, or planner semantics were changed.
