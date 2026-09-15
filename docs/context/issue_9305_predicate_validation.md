# Issue #9305: trace-predicate validation contract

Issue: <https://github.com/ll7/robot_sf_ll7/issues/9305>

Status: `Current` implementation slice. Evidence tier: `diagnostic-only`; the
retained production-trace corpus remains unavailable.

## What this slice provides

`robot_sf/analysis_workbench/trace_predicate_validation.py` defines
`trace_predicate_validation.v1`, a reusable admission and reporting contract
for the eight IDs in `trace_failure_predicates.v1`. Each case binds its
available trace by repository-relative URI, strict trace schema, source
identity, source commit, and SHA-256. Missing traces are represented as
`unavailable` and cannot carry positive or negative detector labels.

The contract also requires two independent reviewer label maps, an explicit
`adjudicated` or `pending` state, reviewer effort in minutes, and a reason for
each unavailable detector label. Precision/recall counts use only explicit
adjudicated positive/negative labels. Ambiguous, pending, and unavailable rows
remain in the report's exclusion and unavailable ledgers; no majority vote is
inferred.

Similarity stability is kept with the collision-similarity owner. The
`compare_similarity_groupings` helper compares co-membership pairs for the
declared full-identity and planner/map-ablated assignments. It does not claim
that a group is a validated failure family.

## Committed bounded fixture

The committed set and report are under
[`evidence/issue_9305_predicate_validation/`](evidence/issue_9305_predicate_validation/README.md).
It contains three strict tracked analysis-workbench fixtures from the exact
base commit and one missing-input case. Its detector labels have
`label_origin: fixture_projection`; this is intentional. The report exercises
true-positive, true-negative, false-positive, false-negative, disagreement,
pending, threshold-change, and identity-ablation paths, but those values are
contract fixtures rather than empirical measurements.

Planner and seed identity are source-bound for available traces. Map identity
is marked `fixture_annotation` because the current trace-export schema has no
map field. Observed patterns and causal hypotheses are separate fields;
causal hypotheses are reported as not evaluated and never enter a metric.

## Reproduction and boundary

```bash
uv run python scripts/analysis/validate_trace_predicate_evaluation_issue_9305.py \
  --evaluation-set docs/context/evidence/issue_9305_predicate_validation/evaluation_set.v1.json \
  --output-json docs/context/evidence/issue_9305_predicate_validation/report.v1.json \
  --output-markdown docs/context/evidence/issue_9305_predicate_validation/report.v1.md \
  --repo-root . \
  --expected-source-commit 1e86f17e9460c9828f0c1b03cabe87c21da594ce
```

This command performs static validation and report construction only. It does
not run simulation, a campaign, SLURM, or seed-flip mining. The explicit source
commit pin makes stale or cross-base fixture reuse fail closed. The resulting
precision/recall, unavailable rates, reviewer agreement, threshold stability,
and grouping stability are diagnostic fixture mechanics, not retained-corpus
evidence, planner/map conclusions, or causal claims. The smallest next proof
step is to admit a licensed retained corpus through the same contract with
computed detector outputs and independent human labels.
