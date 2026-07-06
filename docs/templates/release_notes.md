# Release Notes Template

Use this template when preparing release notes to explicitly document the reproducibility badge claims.

## Reproducibility and Artifact Badging

This release claims the following reproducibility badge:

**Claimed Level**: `[none | available | functional | reproduced]`

### Reproducibility Metadata

- **Checklist Path**: `docs/context/evidence/[release_tag]/release_reproducibility_checklist.md`
- **Durable DOI / Asset URL**: `[Zenodo DOI link or GitHub release URL]`
- **Source Commit**: `[commit_hash]`
- **Scenario Matrix**: `[matrix_path] (hash: [matrix_hash])`

### Validation Status

- **Setup & Installation Command**: `[setup_command]`
- **Functional Smoke Test Command**: `[smoke_test_command]`
- **Functional Smoke Test Status**: `[passed | failed | not_run]`
- **Headline Reproduction Status**: `[passed | failed | not_run]`
- **Reproduction Tolerances**: `[tolerances_details]`

### Known Nondeterminism and Limitations

`[List any known nondeterminism sources, expected unavailable/degraded rows, or other reproducibility limitations.]`
