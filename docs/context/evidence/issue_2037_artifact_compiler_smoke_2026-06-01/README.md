# Issue #2037 Artifact Compiler Smoke Evidence

This directory is a promoted artifact-compiler smoke bundle generated on June 1, 2026 from
`docs/context/evidence/issue_1023_scenario_horizons_local_full_2026-05-06`.

Validation from the repository root:

```bash
# Run from the repository root:
uv run python scripts/validation/validate_artifact_catalog.py \
  docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01/artifact_catalog.yaml

# Then run from this evidence directory:
cd docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01
sha256sum --check checksums.sha256
```

Claim boundary: diagnostic publication-prep evidence only. This bundle proves the artifact compiler
path on one compact real campaign bundle; it is not standalone benchmark evidence or paper-facing
truth.

See `docs/context/issue_2037_artifact_compiler_smoke.md` for the full command and caveats.
