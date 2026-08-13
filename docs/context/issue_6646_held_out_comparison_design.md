# Issue #6646 — Held-Out Diagnosis Comparison: Infrastructure Slice Design

## Status

**Infrastructure slice**: implemented and test-covered.
**No scientific-result claim** is made by this module.

The committed reference fixture (`docs/context/evidence/issue_6646_failure_diagnosis_reference_fixture.v1.json`)
carries `review_marker=AI-GENERATED NEEDS-REVIEW`.  This marker is not admissible evidence
for an independently reviewed held-out comparison.  The fixture remains blocked until
independent review/adjudication is completed.  The comparison harness rejects fixtures
with pending review markers before evaluation.

## Summary

This document describes the approved, bounded infrastructure slice for comparing two
already-materialized sets of `failure_diagnosis.v1` records against an independently
authored held-out reference fixture.  The comparison harness adds a method-manifest
validation layer, case-alignment gate, fixture review admission check, learned
metric-input projection, and a versioned report API on top of the existing
`evaluate_failure_diagnosis_quality` evaluator.

## Approved claim boundary

- **Fixture-level diagnostic metrics only**: detection (confusion, agreement), onset
  (IoU, midpoint error), failure-type (exact match, macro-F1), severity (exact match,
  macro-F1).
- **No campaign ranking** or benchmark-level comparison is implied.
- **No scientific result claim** is made.  This infrastructure enables future
  evaluation; it does not itself establish learned-method quality.
- Unknown, unavailable, fallback, degraded, and provenance-incomplete cases are
  excluded per metric and retained in `case_comparisons`.

## Deterministic vs. learned metric inputs

The existing `evaluate_failure_diagnosis_quality` evaluator validates `diagnosis_source`
against the deterministic adapter constant.  This creates an asymmetry:

- **Deterministic `failure_diagnosis.v1` inputs** pass through the evaluator unchanged.
- **Learned comparison metric inputs** are projected: canonical deterministic fields
  (`diagnosis_schema_version` and `diagnosis_source`) are stripped so the evaluator
  can compare detection, onset, failure type, and severity. The original markers are
  preserved in the comparison output (`learned_source_projection` at the report level
  and `_learned_source_preserved` per case comparison).
- **Deterministic-source learned records/payloads are rejected** before projection:
  the learned side must not duplicate the deterministic adapter source.  This raises
  `LearnedSourceError`, it is not a metric exclusion.
- Truthful non-deterministic learned records are never relabeled as deterministic and
  never silently lose provenance.

## Fixture review admission

The comparison harness requires independently reviewed reference fixtures.  A fixture
whose normalized, case-insensitive `review_marker` contains any entry in
`REVIEW_PENDING_MARKERS` (e.g. `"AI-GENERATED NEEDS-REVIEW"`, including suffixed review
metadata) is rejected before evaluation via `validate_fixture_review_admission()`.  This
is a fail-closed admission gate.

The current committed fixture carries `review_marker=AI-GENERATED NEEDS-REVIEW` and is
therefore blocked.  Tests use a separate reviewed fixture helper that removes the
pending marker.

## Fail-closed rules

1. **Pending fixture review marker** → `FixtureReviewPendingError` (rejects before
   manifest validation or evaluation).
2. **Missing manifest field** → `MethodManifestError` (rejects before evaluation).
3. **Duplicate case ids** → `CaseAlignmentError` (rejects before evaluation).
4. **Mismatched case sets** → `CaseAlignmentError` (rejects before evaluation).
5. **Invalid reference fixture** → `FailureDiagnosisError` (from existing validator).
6. **Deterministic-source learned records** → `LearnedSourceError` (rejects before
   projection and evaluation).  The learned side must not duplicate the deterministic
   adapter source.
7. **Missing or empty learned source markers** → `LearnedSourceError` rejects the
   learned method before evaluation; source provenance is never synthesized.
8. **Malformed or metric-incomplete learned inputs** → The evaluator produces
   explicit unavailable/excluded rows per metric; it does **not** promote a
   malformed learned input into a deterministic record.  Unavailable, unknown,
   fallback, degraded, and provenance-incomplete cases remain visible in
   `case_comparisons` and are excluded from metric denominators.
9. **Payload shape ambiguity** → A bare mapping containing a case literally named
   `"records"` is rejected as ambiguous. Only a complete versioned
   `failure_diagnosis.v1` envelope with a non-empty list of record mappings is
   treated as a payload.
10. Provenance is **never synthesised** — empty or missing fields always fail closed.

## Learned metric-input projection

The learned metric-input projection is a contract defined in
`failure_diagnosis_comparison.py`:

- `_LEARNED_PROJECTION_DROP_FIELDS`: canonical fields removed before evaluator
  comparison (currently `("diagnosis_schema_version", "diagnosis_source")`).
- `_project_learned_record_for_evaluator()`: strips drop fields, preserves the
  original values in `_learned_source_preserved` annotation.
- `_project_learned_records()`: applies the projection to all learned records in
  any accepted shape (payload, mapping, or iterable).
- `_attach_learned_source_preservation()`: carries the preserved source marker into
  each learned case comparison after evaluator normalization.
- `_reject_deterministic_source_learned()`: rejects learned records/payloads
  that carry `DIAGNOSIS_SOURCE` before projection.

The comparison report includes a `learned_source_projection` field documenting
the projection description and which fields were preserved.

## Architecture

```
reference_fixture ──┐
                    ├── compare_held_out_diagnoses() ── comparison report (v1)
deterministic ──────┤
records             │
                    │
learned_records ────┘
        │
method_manifest ───┘
```

### Components

| Component | Location | Role |
|---|---|---|
| `MethodManifest` | `failure_diagnosis_comparison.py` | Frozen, slotted dataclass for pinned provenance. |
| `validate_method_manifest()` | `failure_diagnosis_comparison.py` | Fail-closed manifest validation (all 9 fields required). |
| `validate_fixture_review_admission()` | `failure_diagnosis_comparison.py` | Fail-closed admission check for pending review markers. |
| `_reject_deterministic_source_learned()` | `failure_diagnosis_comparison.py` | Rejects learned records with deterministic `diagnosis_source`. |
| `_project_learned_record_for_evaluator()` | `failure_diagnosis_comparison.py` | Projects one learned record into evaluator-compatible format. |
| `_project_learned_records()` | `failure_diagnosis_comparison.py` | Projects all learned records, preserving input shape. |
| `_is_payload_shape()` | `failure_diagnosis_comparison.py` | Distinguishes payload from case-id mapping (rejects `records` ambiguity). |
| `align_held_out_cases()` | `failure_diagnosis_comparison.py` | Rejects duplicate, missing, or mismatched case ids. |
| `compare_held_out_diagnoses()` | `failure_diagnosis_comparison.py` | Primary entry point: validates fixture admission, manifest, rejects deterministic learned source, projects learned records, aligns cases, calls evaluator for each method, emits versioned report. |
| `build_unavailable_comparison_report()` | `failure_diagnosis_comparison.py` | Fail-closed path: explicit `"unavailable"` status with reason. |
| `evaluate_failure_diagnosis_quality()` | `failure_diagnosis.py` | Existing evaluator; reused without duplication. |

### Schema version

`held_out_diagnosis_comparison.v1`

### Report structure

```json
{
  "schema_version": "held_out_diagnosis_comparison.v1",
  "output_status": "available" | "unavailable",
  "output_reason": null | "string",
  "alignment": { "aligned_case_ids": [...], "deterministic_count": N, "learned_count": N },
  "method_manifest": { ... },
  "learned_source_projection": null | {
    "description": "...",
    "preserved_fields": ["diagnosis_schema_version", "diagnosis_source"],
    "preserved_source_fields": ["diagnosis_source"]
  },
  "deterministic_summary": { "metrics": {...}, "case_count": N, ... },
  "learned_summary": { "metrics": {...}, "case_count": N, ... },
  "case_comparisons": [ ... ],
  "deterministic_case_comparisons": [ ... ],
  "claim_boundary": { ... },
  "caveats": [ ... ]
}
```

### Accepted input shapes

The harness accepts three input shapes for records:

1. **Payload**: a complete versioned `failure_diagnosis.v1` envelope with a
   `"records"` key whose value is a non-empty list of mappings. A bare
   `{"records": [...]}` mapping is not a payload because it is ambiguous with a
   case-id mapping.
2. **Case-id mapping**: `{case_id: record, ...}` — a mapping from case identifiers
   to records.  A mapping with a `"records"` key whose value is not a list of
   mappings (e.g. a case literally named `"records"`) is rejected as ambiguous
   when passed to the comparison entry point; provide the complete envelope or
   rename the case.
3. **Iterable**: `[record, ...]` — a sequence of record mappings with case
   identifiers.

## Files added

| File | Purpose |
|---|---|
| `robot_sf/benchmark/failure_diagnosis_comparison.py` | Comparison harness module. |
| `tests/benchmark/test_failure_diagnosis_comparison.py` | Test suite covering happy path, manifest validation, alignment, provenance fail-closed, exclusions, stability, no-model-execution, fixture review admission, learned source projection, payload shape ambiguity, and module contract. |
| `docs/context/issue_6646_held_out_comparison_design.md` | This design document. |

## Test coverage

- Happy path with synthetic frozen outputs (reviewed fixture)
- Manifest validation (missing, empty, whitespace-only, non-mapping, multiple missing)
- Case alignment (perfect, missing-in-learned, missing-in-deterministic, duplicates)
- Provenance fail-closed (missing manifest field, empty field)
- Held-out exclusion declaration
- Unknown/unavailable case exclusions
- Deterministic report stability (idempotent)
- No model execution (source inspection)
- Unavailable report construction
- Payload-shape inputs
- Fixture review admission (pending marker rejected, reviewed accepted, all markers tested)
- Learned source projection (non-deterministic accepted, source preserved, deterministic rejected)
- Payload shape ambiguity (`records` as case name, non-mapping list, empty list, dict, valid payload)
- Module contract (__all__ exports, new error classes)</think>## Next gate for real model output

1. Independent review and adjudication of the reference fixture (remove the pending marker).
2. A pinned method manifest with non-empty provenance fields (method/model id,
   revision, prompt digest, decoding settings, input schema, output artifact digest,
   held-out exclusion declaration, non-deterministic source marker).
3. Execution of the model on the same held-out case set producing frozen output.
4. Independent fixture review and adjudication of the learned output.
5. The comparison harness is then called with the frozen learned records.
