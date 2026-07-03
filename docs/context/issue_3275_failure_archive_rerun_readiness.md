# Issue #3275 Failure Archive Rerun Readiness

This note records the metadata-only gate for rerunning an adversarial proposal model on a separate certified failure archive.

## Boundary

The check is readiness/leakage evidence only. It does not run benchmark campaigns, submit compute jobs, publish artifacts, or claim adversarial proposal models improve failure discovery.

## Implementation

- `robot_sf/benchmark/failure_archive_rerun_readiness.py` loads source archive and rerun archive inputs, checks archive-ID leakage, records family/seed overlap provenance, blocks missing overlap metadata (`archive_id`, scenario family, `candidate.scenario_seed`), blocks duplicate archive IDs within either archive, requires top-level archive lineage in `config.source_manifests`, blocks shared source-manifest lineage across source and rerun archives, requires passing certification metadata on source and rerun archive entries, and requires valid null-test prerequisite metadata tied to the checked source/rerun archive SHA-256 pair.
- `scripts/validation/check_failure_archive_rerun_readiness.py` exposes the check as a fail-closed JSON CLI, including optional `--null-test-prerequisites` input.
- Diagnostic-only rerun reports cap the verdict at `diagnostic_only` rather than `ready`.

## Consolidated closure packet

The many bounded readiness guards above accumulated into a stream of micro-checks
that a reviewer had to re-derive by hand. `robot_sf/benchmark/failure_archive_rerun_closure.py`
consolidates that pair gate into one durable **closure packet** (schema
`failure_archive_rerun_closure_packet.v1`). It adds no new gate: it renames the
pair verdict into a single rerun-facing `disposition` (`ready_for_rerun`,
`fail_closed_blocked`, or `diagnostic_only`), carries the consolidated blocker
list, and annotates it with a deterministic `next_empirical_action` (the first
matching blocker category selects the guidance). `scripts/adversarial/produce_rerun_closure_packet.py`
exposes it as a fail-closed CLI with exit codes `0`/`2`/`3` matching the
disposition; a missing or malformed archive input fails closed rather than
substituting a synthetic fixture.

Real-archive evidence (fail-closed blocker) is recorded under
`docs/context/evidence/issue_3275_rerun_closure_2026-07-03/`: running the packet
on the two real smoke archives (`issue_1502` source, `issue_1501` rerun) blocks
on archive-ID/scenario/seed overlap, missing certification metadata on both
sides, and absent null-test prerequisites, and prints the disjoint-archive next
action. The real disjoint certified rerun with independent planner-execution
outcomes remains the open issue #3275 contract.

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
- missing, absent, invalid, or archive-pair-mismatched null-test prerequisite metadata blocking readiness;
- diagnostic-only rerun outputs staying diagnostic-only.

Closure-packet tests (`tests/adversarial/test_rerun_closure_packet.py`) additionally lock the
consolidation contract: disposition mapping, consolidated blockers, deterministic next-action
selection per blocker category, fail-closed behavior on a missing real archive, and CLI exit codes.
