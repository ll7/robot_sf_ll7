# Issue #2034 Platformization Roadmap (2026-06-01)

Status: current roadmap, June 1, 2026.

Related issue: [#2034](https://github.com/ll7/robot_sf_ll7/issues/2034)

## Purpose

Recent June 2026 merges added enough tooling that Robot SF now has a recognizable platform layer:
artifact publication, trace inspection, backend contracts, SLURM closeout, external data staging,
and repository layout cleanup are connected workflows rather than isolated utilities. This roadmap
maps what is usable now, what remains prototype-only, and what needs validation before it supports
benchmark or paper-facing claims.

This note is a routing surface. It does not add new benchmark evidence, planner-quality evidence,
or paper-facing claims.

## Capability Map

| Capability | Current status | Canonical entry point | Evidence boundary |
| --- | --- | --- | --- |
| Artifact compiler and publication bundle | Usable for diagnostic publication-prep; paper-facing use still needs durable release or DOI-backed bundle proof. | `uv run python scripts/tools/compile_benchmark_artifacts.py`; `uv run python scripts/tools/benchmark_publication_bundle.py export`; [issue_2040_artifact_publication_workflow.md](issue_2040_artifact_publication_workflow.md); [../benchmark_artifact_publication.md](../benchmark_artifact_publication.md) | Compiler outputs and compact evidence bundles are diagnostic until promoted with checksums, release asset or DOI pointer, and visible fallback/degraded caveats. |
| Canonical table export | Usable when `planner_rows` are extracted from `reports/campaign_summary.json`. | `uv run robot_sf_bench export-canonical-table`; [../benchmark_camera_ready.md](../benchmark_camera_ready.md); [issue_2040_artifact_publication_workflow.md](issue_2040_artifact_publication_workflow.md) | Formats existing rows and writes provenance sidecars. It does not recompute metrics or upgrade failed, fallback, degraded, or `not_available` rows. |
| Real trace viewer | Usable as diagnostic visualization after the #2038 smoke. | `uv run python -m robot_sf.render.trace_viewer`; `uv run python scripts/tools/build_simulation_trace_export.py`; [issue_2038_real_trace_viewer_smoke.md](issue_2038_real_trace_viewer_smoke.md); [../debug_visualization.md](../debug_visualization.md) | Proves trace rendering and browser visibility for one generated real-environment trace. It does not prove metric correctness, map-geometry parity, planner quality, or benchmark success. |
| Trace annotations and reports | Usable for qualitative trace review and fixture-backed reporting. | `scripts/tools/render_trace_report.py`; `robot_sf/analysis_workbench/trace_annotation.py`; `tests/fixtures/analysis_workbench/trace_annotation_set_v1/issue_1962_planner_sanity_open_annotations.json`; [issue_1689_simulation_trace_export_schema.md](issue_1689_simulation_trace_export_schema.md) | `trace_annotation_set.v1` is qualitative analysis-workbench evidence. It must stay tied to the trace id and may not invent benchmark metrics. |
| Backend adapter contract | Usable as a contract and decision surface; alternate backends remain diagnostic unless separately proven. | [issue_2013_backend_adapter_contract.md](issue_2013_backend_adapter_contract.md); [issue_2014_simulator_backend_matrix.md](issue_2014_simulator_backend_matrix.md); `robot_sf/sim/registry.py`; `robot_sf/sim/facade.py` | Backends must fail closed for unsupported semantics, fallback, degraded execution, missing assets, or non-native replay. Adapter-backed output is diagnostic unless a dedicated campaign proves eligibility. |
| SLURM job finalizer and discovery | Usable for local metadata closeout and issue-update drafting; submission and utilization remain host-bound. | `uv run python scripts/tools/slurm_job_finalize.py`; [issue_1894_slurm_job_finalizer.md](issue_1894_slurm_job_finalizer.md); [slurm_job_discovery_2026-05-31.md](slurm_job_discovery_2026-05-31.md); [../dev/slurm_submission.md](../dev/slurm_submission.md) | The finalizer can classify observed jobs and checksum local artifacts. Durable benchmark evidence still needs retrievable artifact URIs and the relevant benchmark policy checks. |
| External data assistant | Usable for license-safe local checks and staging manifests; raw external assets are not redistributed by default. | `uv run python scripts/tools/manage_external_data.py list`; `uv run python scripts/tools/manage_external_data.py check <asset-id>`; [../external_data_setup.md](../external_data_setup.md); [../templates/external_data_audit.md](../templates/external_data_audit.md) | `list`, `explain`, and `check` are local-safe. `stage` needs user-provided licensed source data, and `download` fails closed unless a maintainer-approved source path exists. |
| Root layout structured migration | Usable and current for path expectations after the June 2026 cleanup. | [root_layout_structured_migration_2026-06-01.md](root_layout_structured_migration_2026-06-01.md); [issue_2035_path_reference_audit.md](issue_2035_path_reference_audit.md) | Removed root aliases are intentionally not kept. Historical notes remain provenance only; active docs and scripts should use the new paths. |

## Platform State

Usable now:

- Diagnostic artifact compilation and catalog validation from an existing campaign root.
- Publication-bundle export and checksum verification for selected benchmark run outputs.
- Static trace viewer export, fixture-backed annotation reports, and browser pixel smoke validation.
- Backend adapter contract review and simulator-backend routing decisions.
- SLURM job finalizer closeout for observed jobs.
- External data manifest checks for registered assets.
- Root-layout reference routing after the structured migration.

Prototype-only or validation-needed:

- Treating compiler outputs as dissertation or paper-facing artifacts without a durable release or
  DOI-backed bundle.
- Alternate simulator integrations such as MuJoCo, Webots, Gazebo, Isaac Lab, Habitat, or expanded
  CARLA. Current docs route these to diagnostic spikes or monitor-only status.
- Browser smoke on machines without the Playwright/Chromium dependency.
- External-data `stage` workflows without a local licensed source tree and completed audit.
- SLURM submission and utilization workflows on non-SLURM hosts.

## Next Platform-Stabilization PRs

Active adjacent work:

- Issue #2037 / PR #2048 covers the artifact-compiler end-to-end smoke on one tracked compact
  campaign bundle. Do not duplicate that work; use it as the compiler proof once merged.

Next non-overlapping stabilization PRs:

1. **Add optional map geometry to static trace viewer exports.**
   The real-trace viewer is proven, but current smoke evidence still notes auto-bounds with no SVG
   map geometry, obstacles, or zones. A bounded PR should propagate optional map/geometry metadata
   into the viewer scene and validate it with `tests/render/test_trace_viewer.py` plus the browser
   pixel smoke. Keep the claim diagnostic visualization only.

2. **Wire the SLURM finalizer into canonical closeout docs.**
   `scripts/tools/slurm_job_finalize.py` is usable, but the standard submission docs do not yet
   present it as the post-run closeout step. A docs/workflow PR should update
   `docs/dev/slurm_submission.md`, `docs/dev_guide.md`, and nearby context links while preserving
   the helper boundary: local metadata, fail-closed classifications, and no automatic durable
   benchmark evidence.

3. **Add a rendered-docs command smoke for platform snippets.**
   Validate the artifact-publication and trace-viewer snippets that are meant to be copied from
   rendered Markdown. Scope this to docs and small smoke fixtures so it catches shell quoting,
   environment-export, and checksum-directory drift before a reader does.

## Validation Hooks

Cheap docs validation for this roadmap:

```bash
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_2034_platformization_roadmap.md \
  --path docs/context/INDEX.md \
  --path docs/context/README.md \
  --path docs/context/catalog.yaml
```

Useful targeted command checks when a later PR edits the referenced workflows:

```bash
uv run python scripts/validation/validate_artifact_catalog.py <artifact_catalog.yaml>
uv run pytest tests/render/test_trace_viewer.py tests/validation/test_smoke_threejs_viewer_browser.py -q
uv run pytest tests/analysis_workbench/test_trace_annotation.py -q
uv run pytest tests/tools/test_slurm_job_finalize.py -q
uv run python scripts/tools/manage_external_data.py list
```

## Delegated Scout Evidence

This roadmap was drafted after read-only routed scouts:

- OpenCode Zen mapped command/source entry points and local versus infrastructure-bound validation.
- Gemini summarized the capability matrix and recommended a current `docs/context/` roadmap note.
- Copilot identified next stabilization candidates and existing coverage, including #2037 as the
  artifact-compiler smoke lane already in flight. Its changed-file metadata was contaminated by
  concurrent orchestrator edits, so only the written recommendations were used.

Local validation remains the acceptance source for any PR based on this note.
