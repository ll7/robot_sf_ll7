# Restoring a preserved campaign capsule offline

This walkthrough exercises the post-access lifecycle for one small, packaged
campaign capsule. It restores the synthetic fixture, checks its member inventory
and checksums, validates row identities and cardinality, reads the missingness
ledger and compact report, and queries explicit campaign-to-artifact lineage.

## Run it

From the repository root:

```bash
uv run python examples/advanced/40_restore_campaign_capsule.py --json
```

The default run uses a temporary restore root. To inspect ownership behavior with
an explicit root, provide a path that does not already exist:

```bash
uv run python examples/advanced/40_restore_campaign_capsule.py \
  --output-dir /tmp/robot-sf-offline-capsule --json
```

The example removes the root after verification (or after a validation failure)
only when its exact `.robot_sf_offline_capsule_owned` marker proves ownership.
An existing root is never overwritten, and an unowned root is preserved.

## What is checked

1. The versioned capsule is resolved with the canonical
   `cross_host_conformance_capsule.v1` reader.
2. The restored tree is checked against its `chunk_manifest.v1` member list and
   SHA-256 digests.
3. The canonical strict JSONL reader and episode schema validate two unique rows
   with the declared campaign, source, config, seed, and row status.
4. The missingness ledger and compact report agree with the row inventory.
5. The sanitized lineage index returns one campaign-rooted query containing both
   row artifacts and the report artifact.

Use `--case` with `stale_manifest`, `missing_row`, `duplicate_row`,
`wrong_source_identity`, `wrong_config_identity`, `checksum_mismatch`,
`path_escape`, `unsupported_schema`, or `incomplete_copy` to see stable
fail-closed reason codes.

## Evidence boundary

The fixture is deliberately synthetic, `execution_status: not_run`, and
`evidence_status: diagnostic_only`. The example tests artifact plumbing only. It
does not run simulation or training, calculate scientific statistics, establish
benchmark evidence, or promote/publish a claim. Real private artifacts are not
required. The canonical readers remain the owners of artifact, row, and lineage
semantics; this example only composes them into a bounded offline summary.

Canonical owners: [`chunk_manifest.py`](../../scripts/tools/chunk_manifest.py),
[`aggregate.py`](../../robot_sf/benchmark/aggregate.py),
[`schema_validator.py`](../../robot_sf/benchmark/schema_validator.py),
[`lineage_index.py`](../../scripts/tools/lineage_index.py), and the
[artifact evidence vocabulary](../../docs/context/artifact_evidence_vocabulary.md).
