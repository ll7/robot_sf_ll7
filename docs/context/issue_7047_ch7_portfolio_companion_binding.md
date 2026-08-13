# Issue #7047 Chapter 7 Portfolio Companion Binding

This note records the external portfolio provenance binding for the frozen Chapter 7 evidence
package without changing the admitted package bytes.

The immutable package remains under
`docs/context/evidence/issue_6792_ch7_evidence_package_v1/`. Its `SHA256SUMS`, `manifest.json`,
and `publication/materialization_overlay.json` bytes are not part of the #7047 remediation. The
package claim boundary remains release-cell descriptive evidence only; this binding does not admit
trace-level, matched-start, causal, counterfactual, pooled-arm, or paper-grade claims.

The companion file is
`docs/context/evidence/issue_7047_ch7_portfolio_companion_binding.v1.json`. It binds the two exact
package references to the tracked source config
`configs/analysis/ch7_worked_example_portfolio.v1.yaml` at SHA-256
`2fe0723bbb67eb18d25944b6933575b7c7b5a31836062c0bd56540fe4e3923ec`:

- `manifest.json` pointer `/inputs/portfolio_config/sha256`
- `publication/materialization_overlay.json` pointers `/source_portfolio/path` and
  `/source_portfolio/sha256`

`scripts/tools/lint_evidence_registry.py` loads that canonical companion and validates it
fail-closed against the schema version, exact issue, package document paths, JSON pointers, package
document digests, target path, target tracking state, and target digest. Only after those checks
pass does the linter resolve the two exact #7047 findings:

- `hash_without_artifact_path` on the package manifest portfolio digest
- `uncommitted_artifact_missing_location` on the materialization overlay portfolio reference

Invalid, stale, duplicate, untracked, wrong-document, wrong-pointer, or wrong-digest bindings are
linter errors rather than broad finding-code suppressions.
