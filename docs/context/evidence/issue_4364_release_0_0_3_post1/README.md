# Release 0.0.3.post1 corrected publication evidence

This packet registers the corrected successor bundle for release 0.0.3. It packages the existing
20,160 frozen episodes without simulation and resolves the internal publication contradictions
tracked in issues #5530 and #5580.

Claim boundary: this is a row-preserving release-engineering correction, not a new benchmark or
paper claim. All 14 arms are native or declared adapter executions with no fallback/degraded rows,
but the campaign Social Navigation Quality Index (SNQI) contract still reports `fail` under
warning enforcement. The SNQI consistency pass proves only that the stored field and diagnostics
ordering now use the same execution scalarizer.

## Durable artifact pointer

- Draft release: `https://github.com/ll7/robot_sf_ll7/releases/tag/0.0.3.post1`
- Concept DOI: `10.5281/zenodo.19482025`
- Bundle SHA-256:
  `9bf6ea35a17ce812f0a9c841c3681bc072dcf7ba8c121cbcf05113b8514f4de1`
- Execution commit: `a307ef276d701f8d14dead1aa0513f44ee97c0b0`
- Publication commit: `ded9027d2928512c14bc241397e0ab1d8f586654`

The bundle and full root-checksum log stay out of git and are attached to the draft GitHub release.
`artifact_pointer.json` records their names, hashes, roles, and retrieval URLs.

## Verification

From a fresh extraction, run at bundle root:

```bash
sha256sum -c checksums.sha256
```

Run the merged fail-closed publication gate:

```bash
uv run python scripts/tools/publication_preflight.py \
  --bundle-dir <fresh-extraction>/paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_post1_corrected_publication_bundle
```

Run the SNQI field/diagnostics reconciliation gate:

```bash
uv run python scripts/validation/check_release_snqi_field_consistency.py \
  --bundle <downloaded-bundle.tar.gz> \
  --expected-bundle-sha256 9bf6ea35a17ce812f0a9c841c3681bc072dcf7ba8c121cbcf05113b8514f4de1 \
  --expected-release-tag 0.0.3.post1
```

Compact outputs are preserved in `publication_preflight.json`, `snqi_reconciliation.json`, and
`collision_reconciliation.json`. The publication preflight passes with one expected warning:
the execution and publication commits differ and are allowed by the structured
`provenance.commit_reconciliation` block.

