# Issue #871 Policy Stack Topology Smoke

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/871>

Predecessor notes:

- [Issue #926 Policy Stack V1 Contract](issue_926_policy_stack_v1_contract.md)
- [Issue #1004 Policy Stack V1 Runtime](issue_1004_policy_stack_v1_runtime.md)
- [Issue #871 Policy Stack Proposal Normalization](issue_871_policy_stack_proposal_normalization.md)

## Goal

This slice adds representative topology proof for the existing `policy_stack_v1` runtime. The first
runtime PR proved normal map-runner construction on `planner_sanity_simple`, but that smoke ended at
`max_steps` and did not exercise an issue-596-style topology case.

## What Changed

`tests/planner/test_policy_stack_v1.py` now runs `policy_stack_v1` through `run_map_batch(...)` on
the `corridor_following` scenario from `configs/scenarios/sets/atomic_navigation_minimal_full_v1.yaml`.
The test pins seed `59621`, resolves the scenario map path for inline map-runner use, and asserts:

- one benchmark job is written without failures,
- the episode terminates as `success`,
- `algorithm_metadata.policy_semantics` is `policy_stack_v1_portfolio`,
- planner kinematics report adapter execution,
- kinematics feasibility counts match the number of executed steps,
- planner-runtime diagnostics include native and adapter proposal counts.

## Boundary

This is still experimental portfolio-planner evidence. It does not promote `policy_stack_v1` to a
paper-facing planner, add learned proposals, or prove the full issue-596 matrix. It does close the
previous evidence gap that `policy_stack_v1` lacked a topology-heavy atomic success through the
normal benchmark runner.

## Validation

Targeted smoke:

```bash
uv run pytest \
  tests/planner/test_policy_stack_v1.py::test_policy_stack_runs_atomic_topology_smoke_through_map_runner -q
```

Observed result:

```text
1 passed in 38.93s
```

Focused lint:

```bash
uv run ruff check tests/planner/test_policy_stack_v1.py
```

Observed result:

```text
All checks passed!
```

Focused planner file:

```bash
uv run pytest tests/planner/test_policy_stack_v1.py -q
```

Observed result:

```text
9 passed in 39.27s
```

The topology smoke is a real simulator/map-runner check and currently appears in the repository's
soft slow-test report. It is intentionally retained as the representative #871 proof rather than
being replaced with a mocked runner assertion.
