# Issue #1496 BC-compatible oracle trace collection

Plain-language summary: job 13520 materialized the previously missing `expert_traj_v1.npz`
dataset and registered its private data-plane pointer and checksum. The dataset stays private and is
not committed to Git.

## Registration

- Job: `13520` (`07-issue1496-oracle-trace-collection`).
- Execution commit: `9d65072ecd9d04e2f664a4299665dbff718401d9`.
- Dataset ID: `expert_traj_v1`.
- Private pointer:
  `private-artifact://oracle-imitation/issue1496/oracle-trace-collection/dataset/expert_traj_v1.npz`.
- Dataset SHA-256: `433c797f6e3635f133d7541c9dd9edd3849cdee7b89e23c930b458e463979388`.
- Size: 363,117,931 bytes.
- Dataset schema: valid; per-step observations/actions present.
- Episodes: 12 (`train=6`, `validation=3`, `evaluation=3`).
- Steps: 1,173 total (`train=616`, `validation=201`, `evaluation=356`).
- Strict private-root registry result: `status=valid`, `training_ready=true`.
- Training performed: false.

The public registry records private trace, source-manifest, checksum-inventory, and NPZ pointers.
Strict readiness still requires the authorized private artifact root so the validator can resolve
and hash the bytes; public metadata alone does not bypass that fail-closed check.

## Disposition and honest sufficiency assessment

The materialization blocker reported on #1496 is resolved: a BC-compatible NPZ with per-step
observations/actions now exists and its checksum was independently verified during registration.

The collection is **small for a BC warm-start**. The #1496/#1397 protocol sets split and leakage
requirements but no numeric minimum episode or transition count. Six training episodes provide only
616 training steps, and one training episode contains a single step. That is sufficient for loader,
schema, overfit, and end-to-end smoke checks; it is not a credible basis for the issue's full
BC-warm-start-versus-RL comparison or for a sample-efficiency/final-performance conclusion.

Before committing the predeclared full training budget, run the bounded BC smoke/overfit check and
collect a larger, balanced training set with multiple nontrivial trajectories per scenario. The
full comparison should remain gated until that larger collection passes the same split, checksum,
and private-resolution contracts.

## Evidence boundary

Dataset availability and integrity only. No BC, PPO, DAgger, benchmark comparison, checkpoint, or
paper-facing result is registered here. Evaluation rows remain holdout-only under the existing
split contract.

## Files

- `acceptance.json`: sanitized collection acceptance and dataset pointer.
- `collection_plan.json`: predeclared source, scenarios, seeds, episode IDs, and splits.
- `strict_registry_validation.json`: compute-side strict private-root registry result.
- `registration.json`: external-data-plane provenance, checksums, and sufficiency disposition.
- `SHA256SUMS`: compact bundle integrity manifest.
