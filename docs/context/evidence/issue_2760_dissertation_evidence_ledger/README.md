# Issue #2760 Dissertation Evidence Ledger

Machine-readable JSON ledger for the integrated dissertation evidence status.

## Files

- `ledger.json` - JSON ledger with rows, stale-artifact summary, reuse recommendations, and claim boundaries.

## Schema

`dissertation_evidence_ledger.v2` with fields: `area`, `claim`, `artifact_status`, `evidence_tier`,
`allowed_wording`, `caveat`, `source_issues`, `dissertation_chapter`, `claim_gap`,
`evidence_promotion_path`.

`evidence_promotion_path` is either a string describing the concrete next step required to
promote a diagnostic-only row toward benchmark/paper evidence, or `null` when no credible
promotion path exists. A non-null path does **not** upgrade the row's current classification;
the path must be completed and the evidence tier reclassified before any promotion.

## Validation

```bash
uv run pytest tests/docs/test_dissertation_evidence_ledger.py -q
```
