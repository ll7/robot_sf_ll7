# Planner Adapter Starter Template

Issue: [#1087](https://github.com/ll7/robot_sf_ll7/issues/1087)

This guide is the copy-and-adapt path for adding a small local planner adapter to the Robot SF
map benchmark. It documents the minimum seams a contributor must update and points to a runnable
diagnostic reference adapter that stays aligned with tests.

## Reference Adapter

The minimal in-repo example is:

* adapter class: `robot_sf.planner.socnav.TrivialReferencePlannerAdapter`
* benchmark key: `trivial_reference`
* alias: `reference_adapter`
* tests:
  * `tests/test_socnav_planner_adapter.py`
  * `tests/benchmark/test_map_runner_utils.py`
  * `tests/benchmark/test_algorithm_metadata_contract.py`
  * `tests/benchmark/test_algorithm_readiness_contract.py`

The reference adapter accepts a SocNav structured observation and returns a bounded
`(linear_velocity, angular_velocity)` command. It is deterministic and diagnostic-only; do not use
it as benchmark evidence.

## Minimal Adapter Contract

Create an adapter class that exposes:

```python
class MyPlannerAdapter:
    def __init__(self, config: MyPlannerConfig | None = None) -> None: ...
    def reset(self, *, seed: int | None = None) -> None: ...
    def close(self) -> None: ...
    def diagnostics(self) -> dict[str, object]: ...
    def plan(self, observation: dict[str, object]) -> tuple[float, float]: ...
```

Required behavior:

* `plan()` returns finite `(v, omega)` values in `unicycle_vw` command space.
* Commands are bounded by the adapter config before map-runner projection.
* `reset(seed=...)` is accepted even when the adapter is deterministic.
* `diagnostics()` returns JSON-safe metadata if runtime state should appear in episode records.
* Missing optional dependencies fail closed or return explicit diagnostic status; do not silently
  downgrade benchmark evidence to fallback behavior.

## Files To Update

For a new adapter key, update these surfaces together:

* `robot_sf/planner/<module>.py`: adapter implementation and config builder.
* `robot_sf/benchmark/map_runner.py`: import the adapter and add the `_build_policy()` branch.
* `robot_sf/benchmark/algorithm_readiness.py`: readiness tier, aliases, and opt-in policy.
* `robot_sf/benchmark/algorithm_metadata.py`: category, policy semantics, kinematics contract,
  adapter name, and projection policy.
* `configs/algos/<planner>.yaml`: stable config if the adapter has non-trivial parameters.
* `docs/benchmark_experimental_planners.md`: guardrail status when the planner is testing-only.
* `docs/context/issue_<number>_<topic>.md`: issue-specific decision and validation record.

## Starter Branch Checklist

1. Copy `TrivialReferencePlannerAdapter` and rename the class/config fields for the new planner.
2. Add an explicit benchmark key and aliases in `algorithm_readiness.py`.
3. Add metadata in `algorithm_metadata.py` before any benchmark run can cite the planner.
4. Wire `_build_policy()` through `_build_adapter_policy()` unless the planner has a truly native
   environment-action path.
5. Add direct adapter tests for command shape, bounds, deterministic reset behavior, and error
   handling.
6. Add map-runner dispatch and metadata tests.
7. Run a small smoke with `robot_sf_bench run` on a committed scenario matrix.

## Reference Smoke

The diagnostic reference adapter can be exercised with:

```bash
uv run robot_sf_bench run \
  --matrix configs/scenarios/planner_sanity_matrix_v1.yaml \
  --out output/benchmarks/reference_adapter_smoke/episodes.jsonl \
  --algo reference_adapter \
  --repeats 1 \
  --horizon 300 \
  --workers 1 \
  --no-video \
  --benchmark-profile experimental \
  --algo-config configs/algos/reference_adapter.yaml
```

`configs/algos/reference_adapter.yaml` exists only to opt into the diagnostic/testing-only path.
Do not include `reference_adapter` in benchmark comparison matrices except as a wiring smoke.

