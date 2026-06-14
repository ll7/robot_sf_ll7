# Issue #2767 Benchmark Table Candidates

Status: draft-only synthesis, not benchmark or paper-facing evidence.

This bundle contains conservative benchmark-results table candidates generated from tracked
claim/evidence inputs. It is intended as a review surface for future manuscript planning, not as
manuscript text.

## Contents

- `table_candidates.md`: six draft table candidates requested by issue #2767.
- `summary.json`: source inputs, table IDs, conservative rules, and claim boundary metadata.

## Source Inputs

- `docs/context/evidence/issue_2760_dissertation_evidence_ledger/ledger.json`
- `docs/context/issue_1542_manuscript_claim_evidence_map.md`
- `docs/context/evidence/issue_2799_signalized_runtime/summary.json`
- `docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13/summary.json`

## Regeneration

```bash
uv run python scripts/tools/generate_benchmark_table_candidates.py
uv run pytest tests/docs/test_issue_2767_benchmark_table_candidates.py -q
```

## Claim Boundary

Every table remains `draft_only_not_for_manuscript_use_without_verification`. Diagnostic, stale,
non-claimable, unavailable, fallback, degraded, proxy-only, or missing-denominator evidence weakens
or blocks wording. Missing tracked inputs produce `unavailable` rows rather than invented values.
