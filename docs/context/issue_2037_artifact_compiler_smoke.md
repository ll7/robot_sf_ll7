# Issue #2037 Artifact Compiler Smoke

Status: diagnostic/publication-prep evidence, June 1, 2026.

Related issue: [Issue #2037](https://github.com/ll7/robot_sf_ll7/issues/2037)

## Summary

The benchmark artifact compiler ran end to end on the tracked compact campaign bundle
`docs/context/evidence/issue_1023_scenario_horizons_local_full_2026-05-06`. The run generated the
expected campaign-table variants, planner-status figure, captions, not-available input report,
artifact catalog, and checksum manifest. The generated catalog validates under
`artifact_catalog.v1`.

Promoted evidence bundle:
[docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01/](evidence/issue_2037_artifact_compiler_smoke_2026-06-01/)

This is diagnostic publication-prep evidence only. It proves the compiler path on one compact real
campaign bundle. It does not create a new benchmark result, paper-facing claim, planner-quality
claim, or dissertation table by itself.

## Command

```bash
uv run python scripts/tools/compile_benchmark_artifacts.py \
  --campaign-root docs/context/evidence/issue_1023_scenario_horizons_local_full_2026-05-06 \
  --output docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01 \
  --catalog-id issue_2037_artifact_compiler_smoke
```

Observed output:

```json
{
  "artifact_catalog": "docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01/artifact_catalog.yaml",
  "checksums": "docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01/checksums.sha256",
  "output": "docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01"
}
```

## Generated Outputs

| Output | Promoted path |
| --- | --- |
| Artifact catalog | `docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01/artifact_catalog.yaml` |
| Checksum manifest | `docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01/checksums.sha256` |
| Captions | `docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01/captions.md` |
| Campaign table CSV/Markdown/LaTeX | `docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01/tables/` |
| Planner status figure PNG/PDF | `docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01/figures/` |
| Missing optional inputs report | `docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01/not_available_inputs.json` |
| Copied compact source reports | `docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01/sources/reports/` |

The generated `not_available_inputs.json` records two optional missing source reports:

- `reports/seed_episode_rows.csv`
- `reports/seed_variability_by_scenario.csv`

Those missing optional inputs are caveats for downstream publication prep, not silent zero values.

## Validation

```bash
uv run python scripts/validation/validate_artifact_catalog.py \
  docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01/artifact_catalog.yaml
```

Result:

```text
artifact catalog valid: docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01/artifact_catalog.yaml
```

Additional checks:

```bash
cd docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01
sha256sum --check checksums.sha256
```

## Evidence Boundary

The source campaign bundle is tracked compact evidence, not the full raw benchmark archive. The
compiler output remains a diagnostic publication-candidate stage until the source campaign and
selected artifacts are promoted through a release asset, DOI-backed bundle, or other durable
publication store with checksums. Fallback, degraded, failed, and `not_available` statuses must
remain visible in any downstream table, caption, or manuscript note.
