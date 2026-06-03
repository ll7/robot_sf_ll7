# Hazard And ODD Coverage Summary

Coverage rollup separates metadata-only contract surfaces from executed benchmark evidence. Fallback, degraded, failed, and not_available rows remain caveats and are not success evidence.

- Executed/row-level records read: 21
- Hazard statuses: missing=5
- ODD boundary statuses: excluded=8, partial=2
- Scenario contract statuses: missing=1

## Interpretation Caveats

- `covered` requires at least one non-caveated executed row.
- `partial` preserves fallback, degraded, failed, or not_available row caveats.
- `missing` means metadata maps the category but no executed row represented it.
- `excluded` comes from ODD exclusions or fail-closed scenario certification metadata.
- `unavailable` means an optional metadata surface or campaign table was absent.
