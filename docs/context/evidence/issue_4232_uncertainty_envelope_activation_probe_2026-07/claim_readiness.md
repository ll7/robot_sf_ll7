# Claim Readiness

claim_scope: diagnostic_only
evidence_tier: diagnostic
comparator: envelope_off_alpha_0
baseline: envelope_off_alpha_0
mechanism_activation: see envelope_activation_diagnostics.json
seed_slice_boundary: fixture rows only; no campaign executed by this builder
artifact_provenance: metadata.json and SHA256SUMS
fallback_degraded_limitations: row_status_audit.csv excludes fallback, degraded, not_available, failed, blocked, and diagnostic_only rows from benchmark strength.

Readiness result: not ready for benchmark-strength, conformal, real-world safety, deployment, paper, or dissertation claims.
