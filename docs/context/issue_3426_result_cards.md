# Issue 3426 Result Cards

Issue #3426 adds `scripts/reporting/generate_result_card.py`, a conservative helper for turning
accepted evidence summaries into reusable dissertation material without widening the claim.

The generator writes:

- `result_card.md`
- `result_card.json`
- optional `result_card_table.tex`

The CLI requires an explicit evidence tier, final decision, comparator, command provenance, claim
boundary, metrics, and caveats. It can read common fields from existing summaries, but it fails
closed when the claim boundary, metrics, comparator, command provenance, or caveat context is
missing. Local-only `output/` references are rejected unless the caller explicitly acknowledges a
paired durable pointer or tracked evidence copy.

Examples:

```bash
uv run python scripts/reporting/generate_result_card.py \
  docs/context/evidence/issue_3164_frozen_forecast_policy/summary.json \
  --output-dir output/result_cards/issue_3164 \
  --evidence-tier diagnostic-replay \
  --decision diagnostic \
  --comparator "shared frozen no-forecast replay baseline" \
  --command "uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/evidence/issue_3164_frozen_forecast_policy/summary.json" \
  --latex-table
```

```bash
uv run python scripts/reporting/generate_result_card.py \
  docs/context/evidence/issue_3142_fast_pysf_force_optimization_summary.json \
  --output-dir output/result_cards/issue_3142 \
  --evidence-tier performance-smoke \
  --decision diagnostic \
  --comparator "same command shape baseline versus after optimization" \
  --claim-boundary "Single before/after smoke comparison; diagnostic support only, not a robust speedup claim."
```

This tool does not invent citations, benchmark conclusions, or paper-facing claims. Its output is a
structured bridge from already accepted repository evidence into draftable result-card material.
