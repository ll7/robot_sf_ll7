# Issue #1004: `policy_stack_v1` Minimal Runtime Slice

## Goal

Issue #1004 implements the first runnable `policy_stack_v1` slice under parent issue #871,
following the contract defined in
[Issue #926 Policy Stack V1 Contract](issue_926_policy_stack_v1_contract.md).

The scope is intentionally narrow: prove a config-first planner entry point, two in-repo proposal
sources, explicit availability/status diagnostics, and normal map-runner construction. It is not a
paper-facing benchmark result and does not close the parent portfolio-planner epic.

## Implemented Surface

- Planner module: `robot_sf/planner/policy_stack_v1.py`
- Config: `configs/algos/policy_stack_v1.yaml`
- Map-runner registration: `policy_stack_v1` builds `PolicyStackV1Adapter` with:
  - `goal` as the native proposal source,
  - `risk_dwa` as the adapter proposal source.
- Benchmark metadata/readiness:
  - `baseline_category: classical`,
  - `policy_semantics: policy_stack_v1_portfolio`,
  - `planner_kinematics.execution_mode: adapter`,
  - experimental readiness with explicit opt-in required.

## Diagnostics Boundary

Step diagnostics distinguish these statuses:

- `native`
- `adapter`
- `fallback`
- `degraded`
- `failed`
- `not_available`
- `rejected`

Default `policy_stack_v1.yaml` exercises `native` plus `adapter`. Unit tests also cover failed and
not-available proposal sources, mandatory-source fail-closed behavior, shield fallback
intervention, and JSON-safe diagnostic serialization with `allow_nan=False`.

Fallback and degraded statuses are diagnostic modes, not benchmark-strengthening evidence. A run
that selects fallback or degraded proposals should be reported as a caveat unless a future issue is
explicitly measuring that mode.

## Validation

RED check before implementation:

```bash
rtk uv run pytest tests/planner/test_policy_stack_v1.py -q
```

Result: failed during collection with
`ModuleNotFoundError: No module named 'robot_sf.planner.policy_stack_v1'`.

Focused post-implementation test:

```bash
rtk uv run pytest tests/planner/test_policy_stack_v1.py -q
```

Result: `6 passed`.

Focused lint:

```bash
rtk uv run ruff check robot_sf/planner/policy_stack_v1.py tests/planner/test_policy_stack_v1.py
```

Result: `All checks passed!`

Map-runner smoke:

```bash
LOGURU_LEVEL=INFO rtk uv run python -c 'import json; from pathlib import Path; from robot_sf.benchmark.runner import run_batch; out=Path("output/benchmarks/issue_1004_policy_stack_v1_smoke/episodes.jsonl"); out.parent.mkdir(parents=True, exist_ok=True); summary=run_batch(Path("configs/scenarios/single/planner_sanity_simple.yaml"), out_path=out, schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"), algo="policy_stack_v1", algo_config_path=Path("configs/algos/policy_stack_v1.yaml"), workers=1, horizon=220, dt=0.1, record_forces=False, resume=False, fail_fast=True, benchmark_profile="experimental"); print(json.dumps(summary, indent=2, sort_keys=True))'
```

Result:

- `total_jobs: 3`
- `successful_jobs: 3`
- `benchmark_availability.benchmark_success: true`
- `algorithm_readiness.profile: experimental`
- `planner_kinematics.execution_mode: adapter`
- output: `output/benchmarks/issue_1004_policy_stack_v1_smoke/episodes.jsonl`

Episode outcome caveat: the three smoke episodes terminate at `max_steps`, so the run proves normal
benchmark entry and diagnostics emission, not planner quality or route-completion readiness.
Inspection of the current JSONL output found `rows: 3`, `nonfinite_score_values: 0`, and first-row
planner status counts of `native: 220`, `adapter: 220`, `rejected: 220`.

## Follow-Up Boundary

- Parent #871 remains open for the full portfolio-planner work: richer proposal normalization,
  route/subgoal behavior, representative scenario proof, and any paper-facing promotion.
- Issue #884 can reuse the new status/rejection diagnostics to explain classic merge failures before
  changing policy behavior.
- Fallback/degraded execution must remain a limitation label unless a future task explicitly exists
  to measure that mode.
