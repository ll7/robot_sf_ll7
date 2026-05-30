# Issue #1685 Dummy Learned-Policy Adapter Fixture

Date: 2026-05-30

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1685>

## Goal

Issue #1685 adds a deterministic learned local-policy adapter fixture that exercises the Robot SF
learned-policy boundary without loading a checkpoint, running training, or claiming benchmark
performance.

## Adapter Boundary

- Adapter path: `robot_sf/planner/learned_policy_adapter.py`
- Fixture: `DummyLearnedLocalPolicyAdapter`
- Observation level: `lidar_2d`
- Planner observation mode: `sensor_fusion_state`
- Required deployment inputs: `drive_state` and `rays`
- Action command space: `unicycle_vw`
- Deterministic action: `{"v": 0.25, "omega": 0.0}`

The fixture metadata includes checklist-style `observation_t`, observation field classifications,
split/provenance fields, action contract fields, and per-step logging fields. The
`claim_boundary` is `adapter_fixture_only_not_benchmark_evidence`, and `candidate_registry` keeps
`entry_planned: false`.

The fixture also exposes planner-protocol lifecycle hooks (`step`, `reset`, `configure`, and
`close`) so adapter-boundary smoke tests can use it without pretending it is a promoted benchmark
candidate.

## Fail-Closed Behavior

The adapter raises `LearnedPolicyAdapterContractError` before action emission when a request uses
an unsupported observation level, unsupported action command space, missing required observation
keys, or known forbidden evaluation-time keys such as `future_states`.

## Validation

Red state:

```bash
uv run pytest -q tests/planner/test_dummy_learned_policy_adapter.py
```

Result: collection failed because `robot_sf.planner.learned_policy_adapter` did not exist.

Green state:

```bash
uv run pytest -q tests/planner/test_dummy_learned_policy_adapter.py
```

Result: `8 passed`.

The focused test also calls
`scripts.validation.check_learned_policy_eligibility.validate_learned_policy_eligibility()` on the
adapter metadata and expects no checklist issues.

## Follow-Up Boundary

This fixture is not a model registry entry, durable checkpoint, training proof, benchmark claim, or
learned-policy performance result. Future real learned adapters still need checkpoint provenance,
runtime smoke evidence, artifact promotion decisions, and benchmark-specific validation before any
candidate registry or paper-facing claim.
