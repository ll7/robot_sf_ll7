# Issue #2271 Oracle Imitation Trace Preflight

Date: 2026-06-05

## Scope

This note records the local, no-submit preflight for Issue #2271 and the Issue #1470
oracle-imitation trace collection wrapper. It does not submit SLURM, collect traces, create the
final imitation NPZ dataset, upload artifacts, or prove an imitation policy.

## Preflight Inputs

- Issue: [#2271](https://github.com/ll7/robot_sf_ll7/issues/2271)
- Parent execution lane: [#1470](https://github.com/ll7/robot_sf_ll7/issues/1470)
- Launch packet:
  `configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml`
- Wrapper: `scripts/dev/sbatch_oracle_imitation_traces_issue1470.sh`
- Batch script: `SLURM/Auxme/issue_1470_oracle_imitation_traces.sl`
- Source candidate: `hybrid_rule_v3_static_margin0_waypoint2`
- Exact preflight commit: `e0ca196e63aead9e8491eb7f022d3a0e98c4a126`

## Commands Executed

```bash
uv run python scripts/validation/validate_oracle_imitation_launch_packet.py \
  --config configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml --json
```

Result: `status=valid`, `dataset_id=issue_1397_oracle_imitation_v1`, six scenarios, and twelve
planned manifest episode identifiers across train, validation, and evaluation.

```bash
scripts/dev/sbatch_oracle_imitation_traces_issue1470.sh --dry-run --no-status
```

Result: dry-run only; no SLURM job was submitted. The resolved submission command was:

```bash
sbatch --time=06:00:00 --partition=a30 --qos=a30-gpu \
  --job-name=rsf-1470-oracle-traces \
  --export=ALL,ISSUE1470_SPLIT=train,ISSUE1470_HORIZON=500,ISSUE1470_WORKERS=1 \
  SLURM/Auxme/issue_1470_oracle_imitation_traces.sl
```

The wrapper reported `partition_max=unknown`, `qos_max=none`, and an effective time of
`06:00:00`; an allowed Auxme login node should still recheck live partition policy before removing
`--dry-run`.

## Artifact And Split Expectations

The batch script writes trace outputs under the job-local `ROBOT_SF_ARTIFACT_ROOT` and syncs them
to the configured job `RESULTS_ROOT` on exit. For the default dry-runed wrapper settings, the trace
collector targets:

- split: `train`
- horizon: `500`
- workers: `1`
- trace directory name: `oracle_imitation/issue1470_train_candidate_traces`

The launch packet's durable artifact aliases remain pending and must be replaced with concrete
artifact aliases or run IDs before downstream imitation training treats the collection as an input:

- `wandb-artifact://robot-sf/oracle-imitation/issue_1397_oracle_imitation_v1_manifest:pending`
- `wandb-artifact://robot-sf/oracle-imitation/issue_1397_oracle_imitation_v1_npz:pending`

The tracked dry-run fixture checksum was verified:

```text
eb5ef938d15725ff29a013a196216d093dd549ae58ae805e642c98441777529f  docs/context/evidence/issue_1397_oracle_imitation_launch_packet_2026-05-24/dry_run_dataset_stub.json
```

The split contract remains:

- train seeds: `201, 202, 203, 204, 205, 206`
- validation seeds: `101, 102, 103`
- evaluation seeds: `111, 112, 113`
- validation/evaluation seeds must not leak into train,
- hard-slice recovery examples stay out of evaluation unless predeclared,
- generated trace and dataset manifests need recorded checksums before imitation training starts.

## Issue #2441 SLURM Follow-Up (2026-06-06)

Issue #2441 submitted the train and validation trace-collection splits from branch
`gse-2441-oracle-traces` at commit `a9679e1a37495b25c1786917fcf2fa7e749d1475`.

- Job `12762` collected the train split on `a30` and completed with exit code `0:0`.
- Job `12763` collected the validation split on `l40s` and completed with exit code `0:0`.
- Compact checksums and summary metrics are tracked in
  `docs/context/evidence/issue_2441_oracle_imitation_traces_2026-06-06/README.md`.

The result is `completed_pending_artifact_promotion`: the trace collector produced the expected
local manifests and JSONL rows, but the raw traces are still local-only `output/` artifacts. The
classic slices are weak enough that this should remain diagnostic dataset-prep evidence, not an
oracle-quality or benchmark claim. Downstream imitation training still needs durable artifact
promotion or a deliberate rerun/revision.

## Current Decision

Issue #1470 has train/validation trace-collection evidence, but it is not closed as durable dataset
evidence yet. The local evidence proves the launch packet validates and the wrapper can collect
trace rows through SLURM. It is not benchmark evidence, not final dataset evidence, and not
learned-policy evidence.

Next smallest proof step: promote the generated train/validation trace manifests and JSONL files to
a durable artifact store with concrete retrieval aliases, or revise/rerun before any imitation
training treats these local outputs as inputs.

Machine-readable summary:
`docs/context/evidence/issue_2271_oracle_imitation_trace_preflight_2026-06-05/summary.json`
