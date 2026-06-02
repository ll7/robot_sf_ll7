# Issue #2040 Artifact Publication Workflow

Status: current workflow guidance, June 1, 2026.

Related issue: [Issue #2040](https://github.com/ll7/robot_sf_ll7/issues/2040)

## Purpose

This note defines the human-facing path from a benchmark campaign report to a table, figure,
caption, catalog row, and dissertation or manuscript inclusion. It connects the existing benchmark
report outputs, artifact compiler, artifact catalog, canonical table exporter, and publication
bundle surfaces without treating local `output/` files as durable proof.

## Evidence Tiers

| Tier | Meaning | Acceptable use |
| --- | --- | --- |
| Diagnostic | Local or tracked compact outputs used to inspect a workflow or candidate result. | Debugging, review, issue handoff, and workflow smoke evidence. |
| Camera-ready | A benchmark campaign or release rehearsal that follows the camera-ready contract and records fail-closed row status. | Release rehearsal, benchmark review, and candidate table generation. |
| Paper-facing | Frozen benchmark contract plus durable publication bundle, release artifact, checksums, and caveats. | Manuscript, dissertation, release notes, and externally cited tables or figures. |

Diagnostic compiler output can prove that the publication workflow runs. It cannot by itself prove
planner quality, benchmark success, or paper readiness. Paper-facing language needs the durable
release or publication-bundle proof described in
[docs/benchmark_artifact_publication.md](../benchmark_artifact_publication.md) and
[docs/benchmark_camera_ready_release.md](../benchmark_camera_ready_release.md).

## Workflow

1. Start from a benchmark campaign root.

   The root must contain `reports/campaign_table.csv`. Camera-ready campaigns usually also include
   `reports/campaign_summary.json`, `reports/matrix_summary.json`, scenario breakdowns, SNQI
   diagnostics, and seed or statistical sufficiency reports.

2. Compile publication candidates.

   Canonical command path:

   ```bash
   uv run python scripts/tools/compile_benchmark_artifacts.py \
     --campaign-root output/benchmarks/camera_ready/<campaign_id> \
     --output output/benchmarks/publication_candidates/<campaign_id> \
     --catalog-id <campaign_id>_publication_candidates
   ```

   The compiler copies present report inputs into `sources/reports/`, writes table variants under
   `tables/`, figure candidates under `figures/`, caption text in `captions.md`, an
   `artifact_catalog.yaml`, and `checksums.sha256`. Missing optional inputs are recorded as
   `not_available` rows instead of being silently ignored.

   Output under `output/benchmarks/publication_candidates/` is a local candidate stage. It is useful
   for drafting and review, but it is not the durable source cited by a paper or dissertation until
   selected files are promoted to a release, DOI-backed bundle, tracked compact evidence copy, or
   another durable store with checksums.

3. Inspect the artifact catalog.

   `artifact_catalog.v1` is the handoff between generated files and reusable figure or table
   identities. Each row records source files, generated outputs, checksums, generation command,
   generation commit, caption file, and claim boundary. Validate it with:

   ```bash
   uv run python scripts/validation/validate_artifact_catalog.py \
     output/benchmarks/publication_candidates/<campaign_id>/artifact_catalog.yaml
   ```

4. Export a canonical table when the downstream document needs a stable table contract.

   Use this when a report has the row payload expected by a named table contract:

   ```bash
   export CAMPAIGN_ROOT=output/benchmarks/camera_ready/<campaign_id>

   uv run python - <<'PY'
   import json
   import os
   from pathlib import Path

   reports = Path(os.environ["CAMPAIGN_ROOT"]) / "reports"
   summary = json.loads((reports / "campaign_summary.json").read_text(encoding="utf-8"))
   rows = summary["planner_rows"]
   (reports / "planner_rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
   PY

   uv run robot_sf_bench export-canonical-table \
     --table-id planner_outcome_summary \
     --rows "$CAMPAIGN_ROOT/reports/planner_rows.json" \
     --out-dir "$CAMPAIGN_ROOT/reports/canonical_tables" \
     --source "$CAMPAIGN_ROOT/reports/campaign_summary.json"
   ```

   The exporter writes `csv`, `md`, and `tex` fragments plus a metadata sidecar with source
   checksums, command, commit, selected columns, row count, and generated paths. It formats rows; it
   does not recompute benchmark metrics or reinterpret fallback, degraded, failed, or
   `not_available` status.

5. Promote only the right artifact class.

   For review, a compact evidence copy under `docs/context/evidence/` may be enough. For manuscript
   or dissertation claims, export and publish a durable publication bundle:

   ```bash
   uv run python scripts/tools/benchmark_publication_bundle.py export \
     --run-dir output/benchmarks/camera_ready/<campaign_id> \
     --out-dir output/benchmarks/publication \
     --bundle-name <campaign_id>_publication_bundle \
     --release-tag vX.Y.Z \
     --doi 10.5281/zenodo.<record-id>
   ```

   Publication bundles are candidates for GitHub release, DOI, or other durable external storage.
   They should not be copied wholesale into git. Replace placeholder release and DOI values before
   treating the bundle as paper-facing evidence. From the bundle `payload/` directory, run
   `sha256sum --check ../checksums.sha256` before upload or citation.

6. Cite the table or figure with its boundary.

   A dissertation table should cite the catalog artifact id, source campaign or bundle, checksum,
   generation command, and the claim boundary. If the source campaign has failed, fallback,
   degraded, or `not_available` rows, carry those statuses into the caption or table notes.

## Compiler Evidence Map

| Artifact id | Files | Citation dependency |
| --- | --- | --- |
| `fig_planner_status_summary` | `figures/planner_status_summary.{png,pdf}` | Use only with `captions.md`, `checksums.sha256`, `artifact_catalog.yaml`, and the source campaign or bundle pointer. |
| `tab_campaign_table` | `tables/campaign_table.{csv,md,tex}` | Use only when fallback, degraded, failed, and `not_available` statuses remain visible in the table or caption. |
| `tab_not_available_inputs` | `tables/not_available_inputs.md`, `not_available_inputs.json` | Use to explain absent optional report inputs; do not treat absent inputs as zero-valued evidence. |

## Example Smoke

The cheapest local smoke for this issue used an existing tracked compact campaign evidence bundle:

```bash
uv run python scripts/tools/compile_benchmark_artifacts.py \
  --campaign-root docs/context/evidence/issue_1023_scenario_horizons_local_full_2026-05-06 \
  --output output/issue_2040_artifact_compiler_smoke \
  --catalog-id issue_2040_smoke_publication_candidates
```

Observed output:

```text
output/issue_2040_artifact_compiler_smoke/artifact_catalog.yaml
output/issue_2040_artifact_compiler_smoke/checksums.sha256
```

This example proves the compiler path on tracked compact evidence. It is diagnostic workflow
evidence, not a new paper-facing benchmark claim.

## Cross-References

- [docs/artifact_catalog.md](../artifact_catalog.md): `artifact_catalog.v1` schema and validator.
- [docs/benchmark_artifact_publication.md](../benchmark_artifact_publication.md): durable bundle
  and publication policy.
- [docs/benchmark_camera_ready.md](../benchmark_camera_ready.md): camera-ready campaign outputs and
  canonical table exporter.
- [docs/benchmark_release_protocol.md](../benchmark_release_protocol.md): release manifest and
  required-artifact checks.
- [docs/context/artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md): evidence-tier
  vocabulary and local `output/` boundary.
