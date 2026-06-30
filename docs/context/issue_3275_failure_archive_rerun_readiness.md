# Issue #3275 Failure Archive Rerun Readiness

This note records a metadata-only gate for rerunning the adversarial proposal model on a separate certified failure archive.

## Boundary

The check is readiness/leakage evidence only. It does not run benchmark campaigns, submit compute jobs, publish artifacts, or claim that adversarial proposal models improve failure discovery.

## Implementation

- `robot_sf/benchmark/failure_archive_rerun_readiness.py` loads a source archive and rerun archive, checks archive-ID leakage, records family/seed overlap provenance, requires certification metadata on rerun archive entries, and can validate supplied null-test prerequisite metadata.
- `scripts/validation/check_failure_archive_rerun_readiness.py` exposes the check as a fail-closed JSON CLI, including optional `--null-test-prerequisites` input.
- Diagnostic-only rerun reports cap the verdict at `diagnostic_only` rather than `ready`.

## Validation

Focused tests cover:

- disjoint certified archives passing the metadata gate;
- overlapping archive IDs blocking leakage;
- missing rerun certification metadata blocking readiness;
- complete null-test prerequisite metadata staying ready;
- missing or invalid null-test prerequisite metadata blocking readiness;
- diagnostic-only rerun outputs staying diagnostic-only.
