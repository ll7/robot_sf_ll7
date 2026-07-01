# Issue #3275 Failure Archive Rerun Readiness

This note records the metadata-only gate for rerunning an adversarial proposal model on a separate certified failure archive.

## Boundary

The check is readiness/leakage evidence only. It does not run benchmark campaigns, submit compute jobs, publish artifacts, or claim adversarial proposal models improve failure discovery.

## Implementation

- `robot_sf/benchmark/failure_archive_rerun_readiness.py` loads source archive and rerun archive inputs, checks archive-ID leakage, records family/seed overlap provenance, blocks missing overlap metadata (`archive_id`, scenario family, `candidate.scenario_seed`), blocks duplicate archive IDs within either archive, requires top-level archive lineage in `config.source_manifests`, blocks shared source-manifest lineage across source and rerun archives, requires passing certification metadata on source and rerun archive entries, and requires valid null-test prerequisite metadata.
- `scripts/validation/check_failure_archive_rerun_readiness.py` exposes the check as a fail-closed JSON CLI, including optional `--null-test-prerequisites` input.
- Diagnostic-only rerun reports cap the verdict at `diagnostic_only` rather than `ready`.

## Validation

Focused tests cover:

- disjoint certified archives passing the metadata gate;
- overlapping archive IDs blocking leakage;
- duplicate archive IDs within the source or rerun archive blocking ambiguous archive packets;
- missing overlap metadata (`archive_id`, scenario family, `candidate.scenario_seed`) blocking readiness;
- missing or shared top-level archive source-manifest lineage blocking readiness;
- missing or failed source archive certification metadata blocking readiness;
- missing or failed rerun archive certification metadata blocking readiness;
- complete null-test prerequisite metadata staying ready;
- missing, absent, or invalid null-test prerequisite metadata blocking readiness;
- diagnostic-only rerun outputs staying diagnostic-only.
