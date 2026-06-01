# Issue #1961 BC Warm-Start Recoverability

Issue: [#1961](https://github.com/ll7/robot_sf_ll7/issues/1961)
Parent: [#1108](https://github.com/ll7/robot_sf_ll7/issues/1108)
Follow-up rerun issue: [#1977](https://github.com/ll7/robot_sf_ll7/issues/1977)

## Outcome

Classification: `rerun_required`

The historical #1108 / #749 BC warm-start trail is partially recoverable, but not enough to close
the experiment as `artifact_rescue_success`. The job `12472` dataset and BC checkpoint were
preserved as a W&B artifact, while the final PPO fine-tuned checkpoint and required policy-analysis
comparison against `27dbe5xu` and `b60iopxt` were not recovered.

No new training was launched for this classification pass.

## Artifact Table

| Artifact class | Status | Evidence | Next action |
| --- | --- | --- | --- |
| Expert trajectory dataset | `recovered` | W&B artifact `ll7/robot_sf/issue_1108_bc_warm_start_job12472:v0`; `issue_749_b60iopxt_v10_eval_trajectories.npz`, size `3.25 GB`, sha256 `fc62311e2dfb0cbc6742e745e7deb0ae876bda5c588a35101812d3297c902e81`; manifest reported `quality_status=validated`, `episode_count=141`, seeds `[111, 112, 113]`. | Hydrate/checksum from the W&B artifact before any rerun or promotion. |
| BC checkpoint | `recovered` | Same W&B artifact; `issue_749_bc_preinit_v10_policy.zip`, size `41.4 MB`, sha256 `c7ee44796f73c3e58dbf9ba7b006e56452b15d4a4d2dde1bc148f9ea2d826ac1`; BC step `12472.1` completed. | Hydrate/checksum and use as the explicit warm-start input if #1977 continues from preserved state. |
| PPO fine-tuned checkpoint | `missing` | Ledger notes report PPO step `12472.2` timed out around `7,454,720 / 10,000,000` timesteps with `success_rate=0` near the tail. No durable final PPO checkpoint pointer is recorded. | Rerun or explicitly resume/fail-close under #1977. |
| Policy-analysis comparison | `missing` | `docs/context/slurm_issue_batch_status_2026-05-21.md` records comparison status as pending against `27dbe5xu` and `b60iopxt`; no durable comparison report was found. | Run after a durable final PPO checkpoint exists, or record a fail-closed unsuccessful run. |
| Historical local paths | `missing` | This worktree does not contain `output/slurm/issue1108-bcppo-job-12462/`, `output/slurm/12462-issue1108-bc-warm-start.out`, `output/slurm/issue1108-bcppo-job-12472/`, or `output/slurm/12472-issue1108-bc-warm-start.out`. | Do not cite these local paths as evidence. Use the W&B preservation pointer or rerun. |

## Durable References

- Preservation artifact: `ll7/robot_sf/issue_1108_bc_warm_start_job12472:v0`
- Aliases: `issue-1108`, `job-12472`, `preserved-20260525`
- Preservation run: <https://wandb.ai/ll7/robot_sf/runs/19udjzki>
- Source note: `docs/context/worktree_training_preservation_audit_2026-05-25.md`
- Canonical ledger row: `docs/context/slurm_issue_batch_status_2026-05-21.md`
- Historical execution note: `docs/context/issue_1108_bc_warm_start_execution.md`
- Launch packet: `docs/context/issue_749_bc_preinit_ppo_launch_packet.md`

## Recommendation

Close #1108 as superseded by the clean rerun/continuation issue #1977 after this classification
lands. The rerun may use the preserved dataset and BC checkpoint if hydration and checksums match
the recorded values, but it must still produce a durable PPO checkpoint and policy-analysis
comparison before any warm-start conclusion is claimed.

## Validation

Checked on 2026-06-01:

```bash
test ! -e output/slurm/issue1108-bcppo-job-12462
test ! -e output/slurm/12462-issue1108-bc-warm-start.out
test ! -e output/slurm/issue1108-bcppo-job-12472
test ! -e output/slurm/12472-issue1108-bc-warm-start.out
rg -n "issue_749|12462|12472|b60iopxt|27dbe5xu|bc_preinit|warm_start|issue1108|issue_1108" model configs docs scripts memory
uv run wandb artifact ls ll7/robot_sf
```

The W&B artifact listing confirmed `training-artifact ... issue_1108_bc_warm_start_job12472:v0`.
The listing command was stopped after the relevant artifact row appeared to avoid spending more
time enumerating unrelated project artifacts.
