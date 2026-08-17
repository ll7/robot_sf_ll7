# Issue #7197: deterministic failure-diagnosis fixture evaluation

Issue #7197 adds the source-side admission contract for the approved deterministic
diagnosis-quality slice. It runs the existing `failure_diagnosis.v1` adapter and
metric evaluator only after every source case is bound to an immutable manifest entry.

This is implementation and metric-integrity proof. It does not establish diagnostic
accuracy, correction usefulness, campaign ranking, a benchmark result, or a paper claim.

## Admission contract

`failure_diagnosis_fixture_manifest.v1` requires one entry per case with:

- a case and fixture version;
- a source-trace URI and canonical SHA-256 digest of the source predicate payload;
- the `trace_failure_predicates.v1` schema and source predicate identifier;
- independent review, reviewer identity, and adjudication status; and
- explicit exclusion from training and prompt-development data.

The runner rejects pending review metadata, source digest drift, case-set drift, and
source records containing reference labels or review/adjudication metadata. These are
structural blockers, not metric exclusions. The reference fixture's own pending review
marker is also rejected before the adapter runs.

The production reference fixture
`docs/context/evidence/issue_6646_failure_diagnosis_reference_fixture.v1.json` remains
marked `AI-GENERATED NEEDS-REVIEW`; the test-only source bundle under
`tests/benchmark/fixtures/issue_7197_failure_diagnosis/` does not promote it to evidence.

## Code and command

- Admission and adapter runner: `robot_sf/benchmark/failure_diagnosis_fixture.py`
- CLI: `scripts/analysis/evaluate_failure_diagnosis_issue_7197.py`
- Contract fixture: `tests/benchmark/fixtures/issue_7197_failure_diagnosis/`

Run a reviewed source/reference bundle with:

```bash
uv run python scripts/analysis/evaluate_failure_diagnosis_issue_7197.py \
  --manifest path/to/manifest.json \
  --source-predicates path/to/source_predicates.json \
  --reference-fixture path/to/reviewed_reference_fixture.json \
  --output output/diagnosis-quality.json
```

The command exits `2` and emits an explicit `output_status=unavailable` report when
any admission gate fails. It never substitutes, infers, or reruns missing reference
labels.
