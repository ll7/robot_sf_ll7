# Issue #2655 Oracle Imitation Trace-URI Registry

Date: 2026-06-27
Status: Current (implementation slice); lane remains `artifact_retrieval_blocked` until concrete URIs land.

## Scope

This note records the durable trace-URI registry contract and validator added for oracle-imitation
artifacts. It is the registry/validator layer that makes the downstream `training_ready` state
*mechanically* checkable. It does **not** collect traces, copy private traces, publish artifacts,
submit Slurm/GPU jobs, run a benchmark campaign, or claim training readiness — only the local
contract code is added here.

Related prerequisites: #1470, #2441, and the pre-Slurm launch packet (#1397). The launch packet
describes *how a collection run is launched*; this registry describes *whether the resulting raw
traces are durably retrievable*.

## Registry Contract

- Schema version: `oracle-trace-uri-registry.v1`
- Module (canonical owner): `robot_sf/training/oracle_trace_uri_registry.py`
- CLI: `scripts/validation/validate_oracle_trace_uri_registry.py`
- Example entry: `configs/training/ppo_imitation/oracle_trace_uri_registry_example.yaml`
- Tests: `tests/training/test_oracle_imitation_trace_uri_registry.py`

Each registry records, per split (`train`, `validation`, `evaluation`):

- `trace_id` — split/trace identity (unique across the registry),
- `uri` — durable pointer (`wandb-artifact://`, `artifact://`, `s3://`, `gs://`, `https://`); large
  traces stay out of git,
- `sha256` — 64-character checksum (a `pending` sentinel is allowed only while not resolvable),
- `retrieval_status` — `resolvable` | `pending` | `blocked`,
- `local_mirror` (optional) — staged copy verified against `sha256` when present.

## Contract Enforced

Base validation fails closed when a trace is missing a URI; uses a non-durable or worktree-local
`output/` URI; has a malformed `sha256`; omits `sha256` for a `resolvable` trace; declares a
duplicate `trace_id`; uses an unknown `retrieval_status`; or has a `local_mirror` whose bytes do not
match the declared checksum.

## Training-Ready Gate

`training_ready` is `True` only when every required split is present and **every** trace listed
for it is a **concrete** (non-`:pending`) durable URI with a valid checksum and
`retrieval_status: resolvable`. A split that mixes a resolvable trace with a `pending`/`blocked`
one fails closed, because the blocked trace is still a required-but-unretrievable input. The
`--require-training-ready` flag turns this into a fail-closed gate. The lane leaves
`artifact_retrieval_blocked` only when this gate passes for all required traces.

## Validation

```bash
uv run python scripts/validation/validate_oracle_trace_uri_registry.py \
  --config configs/training/ppo_imitation/oracle_trace_uri_registry_example.yaml --json
# add --require-training-ready to fail closed until the traces are resolvable.
uv run python -m pytest tests/training/test_oracle_imitation_trace_uri_registry.py -q
```

The checked-in example is intentionally not training-ready (all splits `pending`), documenting the
current blocked state without overclaiming.

## Evidence Boundary

Local `output/` traces are not durable training proof unless represented by this registry (or
another durable artifact pointer). No benchmark/training claim is made here.
