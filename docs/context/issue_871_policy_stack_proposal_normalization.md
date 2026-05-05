# Issue #871 Policy Stack Proposal Normalization

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/871>

Predecessor note:

- [Issue #1004 Policy Stack V1 Runtime](issue_1004_policy_stack_v1_runtime.md)

## Goal

This slice tightens the `policy_stack_v1` proposal boundary after the first runnable runtime PR.
Child proposal sources must now return finite two-value commands inside the configured linear and
angular speed bounds before they can enter risk scoring.

## Decision

`PolicyStackV1Adapter` normalizes every command-producing proposal through a common helper:

- malformed command shapes become `rejected`,
- non-finite commands become `rejected`,
- commands outside `max_linear_speed` or `max_angular_speed` become `rejected`,
- rejected mandatory sources fail closed just like failed or unavailable mandatory sources.

Rejected proposal diagnostics now contribute to both `proposal_status_counts["rejected"]` and the
step-level `rejected_count`, so episode diagnostics reflect invalid proposals instead of silently
scoring them.

## Scope Boundary

This is not the full #871 portfolio planner. It does not add new proposal families, route/subgoal
rebasing, learned scorers, topology-heavy proof, or paper-facing benchmark promotion. It only
hardens the existing minimal `goal` + `risk_dwa` runtime slice against invalid child commands.

## Validation

TDD evidence for this branch:

```bash
uv run pytest tests/planner/test_policy_stack_v1.py -q
```

The RED run failed because invalid Risk-DWA commands were still reported as `adapter` proposals.
After implementation, the focused planner test file passed with `8 passed`.

Focused lint:

```bash
uv run ruff check robot_sf/planner/policy_stack_v1.py tests/planner/test_policy_stack_v1.py
```

Result: `All checks passed!`
