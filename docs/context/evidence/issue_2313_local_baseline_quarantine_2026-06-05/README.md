# Issue #2313 Local Baseline Quarantine Evidence

This bundle preserves the compact planner output for quarantining the seven absent local-only
baseline model artifacts classified by Issue #2277.

- `summary.json`: sanitized compact summary derived from
  `scripts/tools/plan_model_artifact_promotion.py scan --json`; direct ignored artifact paths are
  intentionally omitted from tracked evidence.
- Claim boundary: metadata/provenance guidance only; not benchmark evidence and not an artifact
  publication.
- Result: all seven rows are `availability=unavailable`; two are
  `unavailable_recover_or_retire`, and five are `unavailable_retire_or_rewrite`.
