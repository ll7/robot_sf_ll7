# Issue #9305 predicate-validation fixture

Status: `diagnostic-only`; retained production traces are **unavailable**.

This directory contains the bounded, versioned evaluation-set contract and a
report generated from it. The set references three schema-valid tracked
`simulation_trace_export.v1` fixtures and one explicit missing-input case. The
detector and reference labels are hand-authored fixture projections so that
true-positive, true-negative, false-positive, false-negative, ambiguous, and
pending-review paths can be checked. They are not retained-corpus evidence.

The contract is fail-closed on:

- exact source commit, strict trace schema, repository-relative URI, and file SHA-256;
- complete labels for all eight predicates and a reason for every unavailable label;
- at least two independent reviewers, explicit adjudication, and preserved disagreement;
- complete threshold-variant case coverage; and
- complete full-identity and planner/map-ablated grouping partitions.

Map IDs in these fixtures are explicit annotations because
`simulation_trace_export.v1` does not carry map identity. Planner IDs and seeds
are source-bound when a trace is available. `causal_hypotheses` is a separate
field and is not used by any metric.

## Reproduce the diagnostic report

From the repository root:

```bash
uv run python scripts/analysis/validate_trace_predicate_evaluation_issue_9305.py \
  --evaluation-set docs/context/evidence/issue_9305_predicate_validation/evaluation_set.v1.json \
  --output-json docs/context/evidence/issue_9305_predicate_validation/report.v1.json \
  --output-markdown docs/context/evidence/issue_9305_predicate_validation/report.v1.md \
  --repo-root . \
  --expected-source-commit b8811949e087fabae6c6b57656bd7090d92e43ac
```

The command performs static contract and file-integrity checks only. It does
not start a simulator, campaign, SLURM job, or seed-flip mining run. The exact
current-base pin is intentional: a future source-commit mismatch fails closed. A future
retained corpus must replace the fixture source kind and label origin only
after access, provenance, licensing, independent review, and adjudication are
accepted; this fixture must not be silently upgraded.
