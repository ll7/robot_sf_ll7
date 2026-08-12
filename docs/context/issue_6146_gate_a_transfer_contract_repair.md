# Issue #6146 Gate A transfer-contract repair

## Plain-language summary

This note records the **capability-only Gate A repair** for the downstream
activation issue #6146. Issue #5303 step 2b freezes a promotion-capable design
and the `#6145` terminal result schema, but it explicitly does **not**
authorize the `#6145` campaign or any transfer work. Issue #6146 is the Gate A
boundary that checks whether a future `#6145` result is structurally allowed to
activate downstream transfer analysis.

This repair is **check-only**: it adds typed data classes, immutable transfer
rows, candidate-clustered uncertainty, and a side-effect-free activation
checker. It does not run planners, searches, replays, confirmations, SLURM
jobs, or empirical outcome generation/import. Gate B remains responsible for
the full semantic promote: byte-level hash verification, `>= 5` admitted
candidates, all required lineage gates, and every counted-weak-point gate.

## Evidence boundary

`capability_only`. The changed surfaces describe *what the transfer contract
requires* and *how to check a `#6145` terminal result for structural
activation*. They are not evidence that any weak point transfers, that any
planner is robust, or that the `#6145` campaign has produced a promotable
result.

- Parent promotion campaign: #6145 (not authorized here).
- Downstream activation issue: #6146.
- Prerequisite promotion-capable preregistration: #6861 / #5303 step 2b.
- Entry gate: merged #6139 corrected recertification.

## What Gate A implements

### Changed surfaces

- `robot_sf/adversarial/transfer_matrix.py`:
  - New schema `adversarial_transfer_matrix.v2`.
  - `CertifiedConfig` gains `primary_mechanism`.
  - `PlannerEval` gains `mechanism` and `eval_seed`.
  - New `TransferRow` (candidate x evaluated-planner x fresh-seed).
  - New `CandidateCluster` (per-candidate uncertainty with explicit
    denominators).
  - New `CapabilityRanking` (no `minimax_regret` claim).
  - `select_certified_configs(..., eligible_only=True)` rejects `stress_only`,
    `fallback`, `degraded`, `unavailable`, `duplicate`, `pre_correction`,
    malformed, and lineage-incomplete rows.
  - `build_gate_a_transfer_matrix` builds immutable rows and clusters.
  - `check_issue_6145_activation` side-effect-free structural activation
    checker.
  - Capability-only report text replaces the historical minimax/regret wording.

- `tests/adversarial/test_transfer_matrix_issue_5303.py`:
  - Regression tests for stress-only rejection, excluded row classes,
    opposite mechanism, repeated/missing seeds, misleading ranking,
    missing lineage, and closure-without-promote.

### Frozen activation contract (`check_issue_6145_activation`)

A `#6145` terminal result may structurally activate downstream work only when
all of the following pass:

- `schema_version` == `issue_5303_search_promotion_result.v2`.
- All required fields are present:
  `schema_version`, `decision`, `contract_sha256`, `execution_commit`,
  `admitted_candidate_count`, `candidate_manifest_sha256`,
  `evidence_packet_sha256`.
- `decision` is one of `promote | stop | inconclusive` and equals `promote`.
- `contract_sha256`, `candidate_manifest_sha256`, and
  `evidence_packet_sha256` are well-formed 64-hex SHA-256 strings.
- When `expected_contract_sha256` is supplied, `contract_sha256` matches it.
- `execution_commit` is a 40-hex git SHA.
- `admitted_candidate_count` is an integer `>= 5`.

Issue closure alone never activates anything. A `stop` or `inconclusive`
decision fails closed.

## Gate B remains fail-closed

Gate A is intentionally **structural only**. Gate B must still verify:

- Byte-level SHA-256 of every referenced artifact against the actual files.
- At least five admitted candidates pass the frozen eligibility and lineage
  gates from `issue_5303_search_promotion_contract.v2`.
- All seven counted-weak-point gates (certification, deterministic replay,
  4-of-5 target failure, 4-of-5 same mechanism, 4-of-5 neutral reference
  success, second execution context, no excluded row class).
- The exact cluster-level permutation inference and both null tests.

No transfer-matrix build, report, or downstream issue state change may occur
until Gate B passes.

## Remaining acceptance gaps

- The current `TransferRow` supports one fresh seed per config/planner pair,
  matching the existing standard-seed protocol. Multi-seed confirmation rows
  will extend `eval_seed` to a set or list without breaking the schema.
- Byte-level hash verification of `candidate_manifest_sha256` and
  `evidence_packet_sha256` is delegated to Gate B; `check_issue_6145_activation`
  only checks hash well-formedness and optional contract-hash matching.
- Real candidate-roster binding and final `#6147` manifest freeze are outside
  this Gate A slice.

## Validation

```bash
uv run pytest tests/adversarial/test_transfer_matrix_issue_5303.py -v
uv run ruff check robot_sf/adversarial/transfer_matrix.py tests/adversarial/test_transfer_matrix_issue_5303.py
uv run ruff format robot_sf/adversarial/transfer_matrix.py tests/adversarial/test_transfer_matrix_issue_5303.py
```
