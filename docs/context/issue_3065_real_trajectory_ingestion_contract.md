# Issue #3065 — Real-trajectory ingestion and artifact-staging contract

**Status**: Contract slice implemented (schema + fail-closed preflight + example template + tests).
Local, dataset-agnostic, no external data staged.

**Issue**: [#3065](https://github.com/ll7/robot_sf_ll7/issues/3065) · **Parent**: #3057

## What this is

A reusable, **bring-your-own-dataset (BYO)** contract for staging real pedestrian/agent trajectory
data. It follows the maintainer decision (issue-audit 2026-06-22): the repository never commits or
hosts raw external data; it tracks only a manifest (metadata, license acknowledgment, retrieval
instructions, the canonical conversion shape, checksums, split naming, and a durable pointer). A
contributor points the tooling at trajectory data they have the rights to use; license compliance is
the supplier's (recorded via an acknowledgment field), not a project-held license.

This specializes the **External artifact pointer** category in
[`artifact_evidence_vocabulary.md`](artifact_evidence_vocabulary.md) rather than creating a parallel
evidence system, mirroring the *Learned-Policy Artifact Manifests* pattern in that doc.

## Components

| Path | Role |
| --- | --- |
| `robot_sf/data_ingestion/schemas/real_trajectory_ingestion_manifest.v1.json` | JSON Schema (manifest shape). |
| `robot_sf/data_ingestion/real_trajectory_contract.py` | Schema load/validate + fail-closed semantic `run_preflight`. |
| `configs/data/real_trajectory_manifest.example.yaml` | Synthetic copy-me template (points at no real data). |
| `scripts/tools/check_real_trajectory_manifest.py` | Preflight CLI (reads only the manifest). |
| `tests/data/test_real_trajectory_contract.py` | Focused tests on synthetic metadata + the template. |

## Contract rules

Structural (JSON Schema): required manifest fields — `source` (url/version/citation/access_date),
`license` (name/posture/acknowledgment, `redistribution: false`), `retrieval`
(instructions/download_url/fail_closed), `checksums` (SHA-256 tree hash, pinned expected hash),
`conversion` (frame rate, units, coordinate frame, timestamp/agent-id/position fields, map context,
missing-data behavior), `splits` (naming + members), `staging` (git-ignored dir, `local_only_raw`,
durable target), `privacy`, `availability`, `benchmark_eligibility`.

Semantic (`run_preflight`, fail-closed):

- BYO posture requires `license.supplier_acknowledgment: true`.
- `staging_dir` must resolve under a git-ignored root (`output/` or `$ROBOT_SF_EXTERNAL_DATA_ROOT`)
  so raw data never enters git.
- `durable_storage_target` must name a real durable boundary (not the disposable `output/` root).
- `benchmark_eligibility: benchmark_candidate` requires `availability: validated`.
- `availability: validated` requires a concrete `checksums.tree_sha256`.

## Relationship to existing owners

- The canonical external-data staging executor is `scripts/tools/manage_external_data.py` (per-asset
  `AssetSpec`, SDD staging via `configs/data/sdd_staging_manifest.yaml`). This contract is the
  **generic, dataset-agnostic BYO** layer that defines the manifest shape and readiness gates; it
  does not duplicate per-asset staging logic. A follow-up issue may wire validated manifests into
  that executor and a converter.
- Durable storage boundary (maintainer Decision A): project-generated derived artifacts → W&B;
  user-supplied raw datasets → local only, referenced by checksum + retrieval instructions.

## Validation

```bash
uv run python scripts/tools/check_real_trajectory_manifest.py \
    configs/data/real_trajectory_manifest.example.yaml
uv run pytest tests/data/test_real_trajectory_contract.py -q
```

## Out of scope / residual risk

- No external dataset is downloaded, copied, or committed; no benchmark/domain-shift consumption.
- No real-world validation claim is made. Downstream claims stay diagnostic-only until real data is
  staged and checksum-validated under a follow-up implementation issue.
- The converter that materializes the canonical frame shape from a source dataset is future work.
