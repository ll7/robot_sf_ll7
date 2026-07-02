# Issue #3948 CBF Safety-Filter Baseline

Status: first-slice implementation, diagnostic only.

## What Landed

- `robot_sf/planner/cbf_safety_filter.py` exposes a pure collision-cone Control-Barrier-Function
  (CBF) filter API plus the existing adapter policy wrapper.
- `robot_sf/benchmark/cbf_safety_filter_runtime.py` binds the CBF filter to map-runner simulator
  state as an opt-in runtime layer.
- Map-runner config accepts `cbf_safety_filter={"enabled": true, "arm_key":
  "cbf_collision_cone_on"}` and records episode metadata plus metrics for intervention,
  infeasibility, and fallback rates.
- Event-ledger provenance records the CBF episode summary when enabled.

## Boundary

- Off by default.
- Mutually exclusive with `safety_wrapper` in this first slice.
- Dynamic Parabolic CBF is scaffolded but raises `NotImplementedError`.
- No metric semantics changed.
- No Slurm, GPU, full benchmark campaign, dissertation claim, or paper-facing safety claim.

## Runtime Config

Predeclared first-slice arm:

```python
{"enabled": True, "arm_key": "cbf_collision_cone_on"}
```

Disabled arm:

```python
{"enabled": False, "arm_key": "cbf_off"}
```

Threshold drift on the enabled arm fails closed; new thresholds require a versioned experimental
arm so benchmark rows remain comparable.

## Validation

- `uv run pytest tests/benchmark/test_cbf_safety_filter_runtime.py tests/planner/test_cbf_safety_filter.py tests/benchmark/test_cbf_safety_filter_policy.py tests/benchmark/test_safety_wrapper_runtime.py tests/benchmark/test_event_ledger.py tests/benchmark/test_map_runner_resume_identity.py -q`
- `uv run ruff check robot_sf/planner/cbf_safety_filter.py tests/planner/test_cbf_safety_filter.py robot_sf/benchmark/cbf_safety_filter_runtime.py robot_sf/benchmark/map_runner_episode.py robot_sf/benchmark/map_runner.py robot_sf/benchmark/map_runner_batch_plan.py robot_sf/benchmark/map_runner_worker.py robot_sf/benchmark/map_runner_identity.py robot_sf/benchmark/event_ledger.py tests/benchmark/test_cbf_safety_filter_runtime.py`
- `git diff --check`

This validation is implementation integrity proof, not benchmark-strength evidence.
