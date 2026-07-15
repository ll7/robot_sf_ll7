# Issue #2655 Oracle Imitation Trace-URI Registry

Date: 2026-07-15
Status: Current; recovered-original train, validation, and evaluation artifacts are registered.

## Scope

This registry makes private oracle-imitation trace availability and integrity mechanically
checkable without publishing the private data. It records stable private-data-plane locators,
checksums, and original-job provenance; the raw trace bytes and host-local roots remain outside
git.

Related issues are #1470 and #2441. The canonical inputs are train job `12911`, validation job
`12763`, and evaluation job `12764`. Evaluation job `12765` is retained as duplicate-run provenance
and is explicitly not the canonical evaluation input.

## Registry Contract

- Schema: `oracle-trace-uri-registry.v1`
- Registry: `configs/training/ppo_imitation/oracle_trace_uri_registry_issue_1470.yaml`
- Canonical validator: `robot_sf/training/oracle_trace_uri_registry.py`
- Command-line interface: `scripts/validation/validate_oracle_trace_uri_registry.py`
- Private locator-registry SHA-256:
  `fd3696e0c920dcf984765d3b2902ffe35cf36635f8fb524262d0c79eea8b37ce`

Each canonical split records its recovered original job, `private-artifact://` trace locator and
SHA-256, source-manifest locator and SHA-256, checksum-inventory locator and SHA-256, and
`retrieval_status: resolvable`. The URI path is relative to a private artifact root supplied at
validation time; the public registry contains no `/home/...` or `/Users/...` path.

## Training-Ready Gate

Metadata validation can run without private-data access, but it reports `training_ready=false` for
private-artifact entries because no bytes were resolved or hashed:

```bash
uv run python scripts/validation/validate_oracle_trace_uri_registry.py \
  --config configs/training/ppo_imitation/oracle_trace_uri_registry_issue_1470.yaml --json
```

Strict validation requires the explicit private data-plane root. It resolves each URI beneath that
root, rejects path traversal, requires every trace, source manifest, and checksum inventory to be a
regular file, and verifies each declared SHA-256 before reporting `training_ready=true`:

```bash
uv run python scripts/validation/validate_oracle_trace_uri_registry.py \
  --config configs/training/ppo_imitation/oracle_trace_uri_registry_issue_1470.yaml \
  --private-artifact-root <private-artifact-root> --require-training-ready --json
```

Strict validation fails closed when the root is absent or invalid, a required file is absent, or
any checksum differs. `retrieval_status: resolvable` alone is insufficient for private artifacts.

## Evidence Boundary

Passing strict validation establishes artifact availability, original-job provenance, and byte
integrity in the configured private store. It does not establish a final NumPy dataset, imitation
training quality, benchmark performance, planner quality, or paper-facing evidence. No collection,
training, benchmark, or Slurm work is part of this registry integration.
