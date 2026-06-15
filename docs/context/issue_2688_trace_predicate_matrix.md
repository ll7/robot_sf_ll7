# Issue #2688 Trace Predicate Benchmark Matrix

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2688>

## Scope

Issue #2688 predeclares a minimal trace-failure predicate benchmark matrix before any predicate
rate interpretation. The versioned proposal lives at:

- [../../configs/benchmarks/issue_2688_trace_predicate_matrix.yaml](../../configs/benchmarks/issue_2688_trace_predicate_matrix.yaml)

The matrix records scenario families, planners, seeds, horizon, inclusion and exclusion rules,
zero-row policy, fallback/degraded handling, required trace fields, and the claim boundary.

## Claim Boundary

This is a proposal and prerequisite, not rate evidence. It does not establish benchmark results,
planner rankings, mechanism frequencies, AMMV effects, dissertation Results claims, or safety
claims.

Predicate table outputs without a matrix remain diagnostic-only and claim-ineligible. Predicate
table outputs with this matrix are still claim-ineligible while the matrix status is `proposed`.
A later evidence-producing issue must promote or revise the matrix before rates can be interpreted.

## Fail-Closed Contract

`scripts/tools/build_trace_failure_predicate_tables.py` now accepts `--matrix`. When no matrix is
provided, aggregate payloads and denominator-health reports mark rate interpretation as not allowed.
When a matrix is provided, missing matrix-required trace fields emit `not_available` predicate rows
instead of being silently treated as negative evidence.

The proposed matrix excludes fallback, degraded, failed, and `not_available` rows from rate
evidence. Fallback/degraded execution remains a caveat or exclusion reason, not a successful
benchmark outcome.

## Validation

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/validation/test_trace_failure_predicates.py -q
scripts/dev/run_worktree_shared_venv.sh -- uv run ruff check robot_sf/analysis_workbench/trace_failure_predicates.py scripts/tools/build_trace_failure_predicate_tables.py tests/validation/test_trace_failure_predicates.py
scripts/dev/run_worktree_shared_venv.sh -- uv run ruff format --check robot_sf/analysis_workbench/trace_failure_predicates.py scripts/tools/build_trace_failure_predicate_tables.py tests/validation/test_trace_failure_predicates.py
git diff --check
```

Expected result: focused predicate tests pass; Ruff and whitespace checks pass.

## Follow-Up

A later benchmark issue should either promote this matrix to `rate_interpretable` with durable trace
exports that satisfy the declared denominator, or revise the scenario/planner/seed set before any
paper-facing rate claim.
