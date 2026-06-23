# Pre-submit evidence-contract preflight

`scripts/validation/preflight_evidence_contract.py` is a **CPU-only pre-submit
gate**: before a SLURM job is submitted, it checks that the current public commit
would actually emit a contract's required evidence-bookkeeping fields. Exit code
`0` means the contract conforms (safe to submit); any non-zero exit means the job
would **fail closed on bookkeeping** and must not be submitted.

## Why

SLURM jobs have burned GPU hours and then failed *closed* because the public code
did not emit a required evidence-contract field. Motivating waste: **issue #1475 /
SLURM job 12913** — the ORCA-residual BC smoke summary was missing
`residual_clipping_rate` and the other required diagnostics, so the stage
classified `missing_required_smoke_evidence` *after* the GPU time was already
spent. All existing evidence tooling
(`scripts/validation/validate_evidence_promotion_gate.py`,
`scripts/tools/slurm_job_finalize.py`,
`scripts/tools/reconcile_slurm_evidence.py`) is **post-run**. This script is the
missing cheap **pre-submit** check.

## It composes existing owners — it is not a new source of truth

The script is a thin orchestrator. It **imports** the required-field list from the
canonical owner and **reuses** the production evidence builder; it never redefines
the contract:

- Required fields come from
  `robot_sf/training/orca_residual_lineage_packet.py`
  (`REQUIRED_ORCA_RESIDUAL_SMOKE_FIELDS`, a public alias derived from the canonical
  `_REQUIRED_DIAGNOSTICS` tuple).
- The evidence block is built by the production
  `scripts/validation/run_policy_search_candidate.py::_attach_orca_residual_smoke_evidence`,
  evaluated against a representative row.
- The built-in representative row is constructed through the **real**
  `GuardedPPOAdapter` + `update_shield_stats`, so it mirrors the post-#1475
  on-main `algorithm_metadata.shield_stats.last_decision.action_adaptation`
  shape — it is not fabricated to trivially pass.

A test (`tests/validation/test_preflight_evidence_contract.py`) asserts the
registry's required-field tuple **is** the owner's exported object (identity, not a
copy), proving there is no second source of truth.

## Usage

```bash
# Default built-in representative on-main row:
uv run python scripts/validation/preflight_evidence_contract.py orca_residual_smoke
# Machine-readable report:
uv run python scripts/validation/preflight_evidence_contract.py orca_residual_smoke --json
# Evaluate a real captured row instead of the synthetic one:
uv run python scripts/validation/preflight_evidence_contract.py orca_residual_smoke --row rows.json
```

No GPU, no SLURM, no network, no training. The report captures the git HEAD SHA so
the result is pinned to a commit.

## Registry shape (how to add a contract)

The registry is `_CONTRACT_REGISTRY: dict[str, _ContractSpec]`. Each `_ContractSpec`
declares:

| field | meaning |
| --- | --- |
| `contract_id` | command-line identifier |
| `required_fields` | **imported from the canonical owner** (no second copy) |
| `owner` | human-readable pointer to the module to fix on failure |
| `build_evidence` | `(row) -> dict`, reusing the production builder; returns the evidence block with `missing_required_fields` |
| `representative_row` | `() -> dict`, a synthetic row mirroring the real on-main shape |

**Adding a second contract is a one-entry change**: append a `_ContractSpec` to the
registry. Nothing else in the file is contract-specific. Keep the `required_fields`
imported from that contract's owner module; if the owner exposes the field list only
as a private constant, add a public alias in the owner module (as was done for
`REQUIRED_ORCA_RESIDUAL_SMOKE_FIELDS`) rather than copying the list here.

## Ops integration (do this)

The **private-ops sbatch wrapper / ops preflight should call this before
`sbatch`** for any job whose evidence rides on a registered contract, e.g.:

```bash
uv run python scripts/validation/preflight_evidence_contract.py orca_residual_smoke \
  || { echo "evidence contract would fail closed; not submitting"; exit 1; }
```

so a job is never submitted if the public commit would emit an incomplete contract.
This is the cheap CPU guard that would have prevented the issue #1475 / job 12913
GPU-hour waste.
