# Issue #1977 BC Warm-Start Cancelled Run Manifest 2026-06-02

Date: 2026-06-02

## Training Run

- Issue: #1977, clean BC warm-start PPO rerun from preserved #1108/#749 artifacts.
- Parent issue: #1108.
- PR: #2021.
- SLURM job: `12689`, `CANCELLED`, `a30`, elapsed `14:47:53`.
- Job name: `gse-1977-bcppo`.
- Node: `auxme-imech172`.
- Job commit: `f3234dc0d835e25f42181839e00ce7e9cba24e69`.
- Submit branch: `issue-1977-bc-warm-start-rerun`.
- Training config:
  `configs/training/ppo_imitation/ppo_finetune_issue_1977_bc_warm_start_rerun.yaml`.
- Local synced root:
  `output/slurm/issue1977-bc-warm-start-ppo-job-12689/`.
- Local stdout:
  `output/slurm/12689-issue1977-bc-warm-start-ppo.out`.

## Outcome

Classification: `cancelled_intermediate_negative_diagnostic`.

The repaired vectorized rerun was infrastructure-healthy but behaviorally unpromising. It reached
`5,361,664` PPO timesteps before maintainer-requested cancellation, with the final logged rollout
success rate still `0`. The run does not produce a final PPO checkpoint, final manifest, or
post-training policy-analysis comparison.

Do not count this as completed #1977 evidence or benchmark evidence. The useful result is a
negative diagnostic: preserving the #1108 BC checkpoint and resuming PPO with the current repair
does not show enough early success to justify spending another long allocation on the same run
shape without redesign.

## Runtime Snapshot

| Field | Value |
| --- | ---: |
| Last logged total timesteps | `5,361,664` |
| Last logged iteration | `119` |
| Last logged fps | `100` |
| Last logged success rate | `0` |
| Last logged updates | `1180` |

Sparse earlier rollout logs briefly reported success rates of `0.01` or `0.02`, but the signal did
not persist and returned to `0` by the tail.

## Artifact Classification

| Artifact class | Classification | Boundary |
| --- | --- | --- |
| Preserved input dataset | `durable-promoted` | Already available through W&B artifact `ll7/robot_sf/issue_1108_bc_warm_start_job12472:v0`; local copy is an ignored cache. |
| Preserved BC checkpoint | `durable-promoted` | Already available through W&B artifact `ll7/robot_sf/issue_1108_bc_warm_start_job12472:v0`; local copy is an ignored cache. |
| Periodic PPO checkpoints from job `12689` | `durable-required-if-used` | Local-only ignored outputs. They are negative diagnostic checkpoints, not final run artifacts. Upload only if a future issue explicitly needs intermediate model inspection. |
| SLURM stdout | `tracked-compact-evidence` | This manifest records the key values and stdout checksum. The raw log remains ignored local output. |
| Final PPO checkpoint | `missing` | The run was cancelled before final save. |
| Policy-analysis comparison | `missing` | No final checkpoint existed to compare against `27dbe5xu` and `b60iopxt`. |

## Checksums

| Artifact | Size bytes | SHA256 |
| --- | ---: | --- |
| `output/slurm/12689-issue1977-bc-warm-start-ppo.out` | `119459` | `78956fff4f2e813c95fb8d17ab1b4c4bbedeb998a8ee9c02124d3f7ed2f412ea` |
| `benchmarks/ppo_imitation/runs/issue_749_bc_pretrain_v10_warm_start.json` | `550` | `944a8acdcfaefc29ca375539271be6f7d6bb065ab06f9e54656b193f4377fbb3` |
| `benchmarks/expert_trajectories/issue_749_b60iopxt_v10_eval_trajectories.json` | `6541` | `46d4418307def11103005aea2b0571236561fd42573216cf9b82ea5d1304a02d` |
| `benchmarks/expert_trajectories/issue_749_b60iopxt_v10_eval_trajectories.npz` | `3248478258` | `fc62311e2dfb0cbc6742e745e7deb0ae876bda5c588a35101812d3297c902e81` |
| `benchmarks/expert_policies/issue_749_bc_preinit_v10_policy.zip` | `41376709` | `c7ee44796f73c3e58dbf9ba7b006e56452b15d4a4d2dde1bc148f9ea2d826ac1` |
| `checkpoints/..._499994_steps.zip` | `138494179` | `bdb3edfc4965a73091a8fc7d2710c64e7e38c485dfdbf73ec4456ecfc2626a4f` |
| `checkpoints/..._999988_steps.zip` | `138494179` | `04c482ed926529140d640a7b23f4d3f4059903ee4b38965ce83f734529ca6975` |
| `checkpoints/..._1499982_steps.zip` | `138494189` | `758b9f6dff28224548790013886e3bb8ea5af6cf683285724a5d3320158e5720` |
| `checkpoints/..._1999976_steps.zip` | `138494189` | `183fc9c570f439d93af9a6fdaf81e5f957de53671398eec693d55169a3e89ce9` |
| `checkpoints/..._2499970_steps.zip` | `138494179` | `7106c75b746eb76709502f989ad22c83f4a9812ae1318da7116ba8b8121602e1` |
| `checkpoints/..._2999964_steps.zip` | `138494180` | `726aa91a5a08a5b857ae8d6d80aa07cfb0ca4e76b2ade7aaee8c912a7ff28797` |
| `checkpoints/..._3499958_steps.zip` | `138494180` | `747f49a9a913143de028d55b772a4777eb892aa3b5495ab0cbda1f827515a960` |
| `checkpoints/..._3999952_steps.zip` | `138494180` | `8f1f002503f52b52230dbe4285c8583a99a98220b8a26eaed9ecaec2d76632be` |
| `checkpoints/..._4499946_steps.zip` | `138494180` | `6e1195e51b1cac034b9d46c8077eeaf95908063f43ca3e56d5469c0f0548d90f` |
| `checkpoints/..._4999940_steps.zip` | `138494190` | `b71643673e52b92d1a5c6cd9344a8725e09defa8dd4a8a4e38d72624b55de37c` |

## Next Action

Stop this run shape. A future #1977-style attempt should either redesign the warm-start objective or
open a narrower intermediate-checkpoint analysis issue. Do not submit another 10M continuation from
these inputs merely to complete the original command.
