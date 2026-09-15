# Issue #9305 predicate-validation fixture

Status: `diagnostic-only`; retained production traces are **unavailable**.

This directory contains the bounded, versioned evaluation-set contract and a
report generated from it. The set references three schema-valid tracked
`simulation_trace_export.v1` fixtures and one explicit missing-input case. The
detector and reference labels are hand-authored fixture projections so that
true-positive, true-negative, false-positive, false-negative, ambiguous, and
pending-review paths can be checked. They are not retained-corpus evidence.

The contract is fail-closed on:

- an explicit source-commit pin, strict trace schema, repository-relative URI,
  file SHA-256, and byte-for-byte equality with the Git blob at that commit;
- complete labels for all eight predicates and a reason for every unavailable label;
- at least two independent reviewers, distinct measured adjudication, and preserved disagreement;
- complete threshold-variant case coverage with unavailable-detector propagation; and
- complete full-identity and planner/map-ablated grouping partitions with closed,
  disjoint feature sets.

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
  --expected-source-commit 1e86f17e9460c9828f0c1b03cabe87c21da594ce
```

The command performs static contract and file-integrity checks only. It does
not start a simulator, campaign, SLURM job, or seed-flip mining run. The exact
current-base pin is intentional: a future source-commit mismatch fails closed. A future
retained corpus must replace the fixture source kind and label origin only
after access, provenance, licensing, independent review, and adjudication are
accepted; this fixture must not be silently upgraded.

Report validation is bound to the admitted evaluation-set payload and recomputes
the complete report projection, including the set digest, source commit, coverage,
metrics, availability, threshold, grouping, review, and observation fields. A
report summary cannot authorize edited derived values by itself.
