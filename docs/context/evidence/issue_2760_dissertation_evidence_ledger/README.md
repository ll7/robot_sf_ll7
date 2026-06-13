# Issue #2760 Dissertation Evidence Ledger

Machine-readable JSON ledger for the integrated dissertation evidence status.

## Files

- `ledger.json` - JSON ledger with rows, stale-artifact summary, reuse recommendations, and claim boundaries.

## Schema

`dissertation_evidence_ledger.v1` with fields: `area`, `claim`, `artifact_status`, `evidence_tier`,
`allowed_wording`, `caveat`, `source_issues`, `dissertation_chapter`, `claim_gap`.

## Validation

```bash
uv run pytest tests/context/test_dissertation_evidence_ledger.py -q
```
