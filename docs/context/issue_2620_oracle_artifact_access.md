# Issue #2620 Oracle-Imitation Artifact Access Audit 2026-06-11

Issue: [#2620](https://github.com/ll7/robot_sf_ll7/issues/2620)
Source issue: [#2441](https://github.com/ll7/robot_sf_ll7/issues/2441)
Parent dataset issue: [#1470](https://github.com/ll7/robot_sf_ll7/issues/1470)
Downstream training issue: [#1496](https://github.com/ll7/robot_sf_ll7/issues/1496)

Status: fail-closed artifact access audit.

## Claim Boundary

This note audits whether the completed Issue #2441 oracle-imitation trace collection is currently
recoverable and durable enough for downstream imitation training. It does not train a policy,
promote raw traces, create a final imitation NPZ dataset, or make benchmark or paper-facing planner
claims.

Tracked compact evidence:
[summary.json](evidence/issue_2620_oracle_artifact_access/summary.json).

## Required Output

```yaml
oracle_imitation_artifact_access:
  source_issue: 2441
  train_pointer:
    status: missing_durable_pointer
    finalizer: finalization_12762.json
    artifact_role: required_combined_train_jsonl
    recorded_sha256: 2f761d09e3944ed0489509ce498499f6fff159afd9bd2eba12b6010170fa8c43
    local_retrieval_status: missing_on_current_machine
    durable_uri: null
  validation_pointer:
    status: missing_durable_pointer
    finalizer: finalization_12763.json
    artifact_role: required_combined_validation_jsonl
    recorded_sha256: acf378ed60b7753805123c80077dcb0a5e04e2a6696907b52fb4eaae6d0d1b72
    local_retrieval_status: missing_on_current_machine
    durable_uri: null
  evaluation_pointer:
    status: missing_durable_pointer
    selected_source: job_12764_a30
    finalizer: finalization_12764.json
    artifact_role: required_combined_evaluation_jsonl
    recorded_sha256: 8d508b22bf8e82783cc8a01d335a6dfacd4c1b47e7a1b80ca0679f157dd83909
    duplicate_job_12765_recorded_sha256: 2a3c1397d4c23962497386ab92c23db243dce7097a2eaa2df9594164a26fdc3c
    local_retrieval_status: missing_on_current_machine
    durable_uri: null
  checksum_manifest: docs/context/evidence/issue_2441_oracle_imitation_traces_2026-06-06/SHA256SUMS
  split_leakage_status: passed_manifest_check
  raw_trace_retrieval_status: blocked
  downstream_training_ready: false
  blocker_if_not_ready: The train, validation, and evaluation raw trace JSONL files and their source manifests need concrete durable retrieval pointers, or a deliberate rerun on a Slurm-capable host that publishes those pointers and verifies the recorded checksums before Issue #1496 starts training.
```

Decision outcome: `artifact_retrieval_blocked`.

## Evidence Checked

The tracked #2441 finalization bundle preserves compact checksums and split routing:

| Split | Job | Rows | Recorded combined JSONL SHA256 | Access status |
| --- | ---: | ---: | --- | --- |
| train | `12762` | 6 | `2f761d09e3944ed0489509ce498499f6fff159afd9bd2eba12b6010170fa8c43` | missing durable pointer; local raw path absent |
| validation | `12763` | 3 | `acf378ed60b7753805123c80077dcb0a5e04e2a6696907b52fb4eaae6d0d1b72` | missing durable pointer; local raw path absent |
| evaluation | `12764` | 3 | `8d508b22bf8e82783cc8a01d335a6dfacd4c1b47e7a1b80ca0679f157dd83909` | missing durable pointer; local raw path absent |
| evaluation duplicate | `12765` | 3 | `2a3c1397d4c23962497386ab92c23db243dce7097a2eaa2df9594164a26fdc3c` | duplicate metric check only; local raw path absent |

Split routing remains valid at the manifest level:

- train seeds: `201, 202, 203, 204, 205, 206`;
- validation seeds: `101, 102, 103`;
- evaluation seeds: `111, 112, 113`.

The issue trail and tracked repository references still describe the raw traces as local-only
ignored artifacts. No W&B, release, S3/GCS, Hugging Face, or equivalent durable retrieval URI was
found for the raw trace JSONL files or source manifests.

## Readiness Decision

Issue #1496 remains blocked. Downstream imitation training may not consume these traces yet because the
raw JSONL files and source manifests are not recoverable through a durable pointer. The current
machine also disallows Slurm submission, so this audit cannot rerun or promote the traces directly.

Next smallest unblocking action: from a Slurm-capable or artifact-capable host, either recover the
recorded raw paths and upload the train/validation/evaluation JSONL plus source manifests to a
durable store, or rerun the bounded collection and publish concrete artifact URIs with matching
checksums.

## Validation

```bash
jq '{artifact_classification, split_leakage_check, analysis, jobs: [.jobs[] | {job_id, split, manifest, combined_jsonl}]}' \
  docs/context/evidence/issue_2441_oracle_imitation_traces_2026-06-06/summary.json
```

```bash
# Each finalizer-recorded required manifest/combined JSONL artifact was checked on this machine.
# Result: all eight required raw files were missing on this machine.
```

```bash
rg -n 'issue_1397_oracle_imitation_v1|issue1470|oracle-imitation|wandb-artifact://|artifact://|s3://|gs://|hf://|huggingface|W&B|wandb' \
  docs configs robot_sf scripts tests
gh issue view 2441 --comments --json body,comments,url,state,title
gh issue view 1470 --comments --json body,comments,url,state,title
gh issue view 1496 --comments --json body,comments,url,state,title
```

Result: tracked compact evidence and issue comments preserve checksums and split routing, but
artifact retrieval remains blocked.
