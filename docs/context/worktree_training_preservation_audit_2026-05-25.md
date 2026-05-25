# Worktree Training Preservation Audit, 2026-05-25

## Goal

Verify the active `robot_sf_ll7.worktrees/` checkouts for generated training outputs that should
be preserved before any future cleanup, worktree pruning, or output compaction.

This audit covers local worktree artifacts only. Files under `output/` remain cache/non-durable
until promoted through the model registry, W&B artifact, GitHub release asset, or another explicit
artifact store.

## Live Queue Snapshot

Checked on 2026-05-25 from the Auxme login node during the audit:

```text
12620|robot-sf-camera-ready-benchmark|RUNNING|l40s|started 2026-05-25T06:03:03
```

Follow-up check after artifact preservation:

```text
12620|robot-sf-camera-ready-benchmark|FAILED|2:0|00:21:29|2026-05-25T06:03:03|2026-05-25T06:24:32
```

Job `12620` is the issue #1353 stress benchmark follow-up, not a training run. It produced a
campaign summary at
`../robot_sf_ll7.worktrees/1353-broader-amv-row-contract/output/benchmarks/issue_1353/issue_1353_paired_stress_broader_baselines_issue1353-stress-rowcontract_20260525_060323/reports/campaign_summary.json`.
The SLURM exit is failed because `socnav_bench` was recorded as `not_available` due to missing
SocNavBench `wayptnav_data`, matching the nominal job's row-contract failure mode.

## Preserve Or Publish Candidates

### Issue #1108 BC Warm-Start PPO

Worktree: `../robot_sf_ll7.worktrees/1108-bc-warm-start-ppo`

Preserved on 2026-05-25 as W&B artifact:

* Artifact: `ll7/robot_sf/issue_1108_bc_warm_start_job12472:v0`
* Aliases: `issue-1108`, `job-12472`, `preserved-20260525`
* Size: `3,289,862,058` bytes
* Preservation run: <https://wandb.ai/ll7/robot_sf/runs/19udjzki>

Preserve the latest validated collection and BC checkpoint from job `12472`:

| Artifact | Size | SHA256 | Decision |
|---|---:|---|---|
| `output/slurm/issue1108-bcppo-job-12472/benchmarks/expert_trajectories/issue_749_b60iopxt_v10_eval_trajectories.npz` | 3.25 GB | `fc62311e2dfb0cbc6742e745e7deb0ae876bda5c588a35101812d3297c902e81` | Preserve/publish if #1108 continues; validated 141-episode dataset. |
| `output/slurm/issue1108-bcppo-job-12472/benchmarks/expert_policies/issue_749_bc_preinit_v10_policy.zip` | 41.4 MB | `c7ee44796f73c3e58dbf9ba7b006e56452b15d4a4d2dde1bc148f9ea2d826ac1` | Preserve/publish if #1108 continues; BC completed before PPO fine-tune timed out. |

Evidence:

* Dataset manifest reports `quality_status=validated`, `episode_count=141`, seeds
  `[111, 112, 113]`, source policy
  `ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200`, and scenario coverage
  `classic_bottleneck_low: 141`.
* `sacct -j 12472` reports collection step `12472.0` completed, BC step `12472.1` completed, and
  PPO fine-tune step `12472.2` timed out after reaching about `7,454,720` timesteps with
  `success_rate=0` near the tail.

Important caveat: the four 3.25 GB dataset files from jobs `12462`, `12463`, `12471`, and `12472`
are same-size but not byte-identical. Their hashes are:

| Job | SHA256 |
|---:|---|
| 12472 | `fc62311e2dfb0cbc6742e745e7deb0ae876bda5c588a35101812d3297c902e81` |
| 12471 | `da667464056e131dc0b5fc9fc17ee7c1f25d1e86fa071e08ad847f6cb2d6c6b5` |
| 12463 | `c95fa10041eb61d284cda8f6fb9328ff81fc15c899dcfc32952db70cb17131c1` |
| 12462 | `7a34468f9737f51c9075b0ea35e7a144e7cebfac2da34c50eb7df5b457d65509` |

Do not delete the older three as "duplicates" unless their metadata differences are explicitly
accepted as irrelevant. The local 21-byte/28-byte files under `output/benchmarks/expert_policies/`
are not useful model artifacts.

### Issue #1024 H500 PPO Retrain

Worktree: `../robot_sf_ll7.worktrees/1024-retrain-learned-planners-h500`

Preserved on 2026-05-25 as W&B artifact:

* Artifact: `ll7/robot_sf/issue_1024_h500_best_checkpoints:v0`
* Aliases: `issue-1024`, `h500`, `preserved-20260525`
* Size: `190,619,004` bytes
* Preservation run: <https://wandb.ai/ll7/robot_sf/runs/v4gaehbc>

The final issue insight is already documented in
`docs/context/issue_1024_h500_ppo_retrain.md`: the H500 retrain is useful evidence but not a
promotion over the registered issue-791 leader. Preserve these only if future #1024/H500 analysis
needs re-evaluation or warm-starting from the checkpoint; otherwise the metrics and W&B run are
enough for the issue decision.

| Artifact | Best eval | SHA256 | Decision |
|---|---|---|---|
| `output/slurm/issue791-reward-curriculum-job-12352/benchmarks/expert_policies/checkpoints/ppo_expert_issue_1024_all_available_h500_schedule_12m_env30_l40s/ppo_expert_issue_1024_all_available_h500_schedule_12m_env30_l40s_best.zip` | step 6,291,456; success `0.900`, collision `0.100`, SNQI `0.134` | `777a12c2f23dab9b4f980737018c880920a3cf3b97ff9bccb1e928c60897a8e4` | Conditional preserve; strongest #1024 checkpoint. |
| `output/slurm/issue791-reward-curriculum-job-12350/benchmarks/expert_policies/checkpoints/ppo_expert_issue_1024_all_available_h500_schedule_12m_env22/ppo_expert_issue_1024_all_available_h500_schedule_12m_env22_best.zip` | step 7,340,032; success `0.900`, collision `0.100`, SNQI `0.088` | `0bb76a330c4ea5d66a1a3f0f397791dd35454e8470e5156074362aaf182439f5` | Lower priority preserve; useful hardware/env-count comparison. |

The registered canonical PPO leader remains
`ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`, which is already a
GitHub release asset under `artifact/models-2026-05-registry-v1`.

### Older Issue #791 Policy-Search Worktree

Worktree: `../robot_sf_ll7.worktrees/origin-2026-04-28-policy_search`

Most large issue-791 policy outputs are superseded by the registered leader and published model
registry artifacts. Do not preserve every intermediate checkpoint. If this worktree is cleaned,
keep only explicit non-registry comparison candidates that still answer a live ablation question:

| Artifact | Best eval | SHA256 | Decision |
|---|---|---|---|
| `output/slurm/issue791-asymmetric-critic-job-12221/benchmarks/expert_policies/checkpoints/ppo_expert_issue_791_asymmetric_critic_promotion_10m_env22_eval_aligned/ppo_expert_issue_791_asymmetric_critic_promotion_10m_env22_eval_aligned_best.zip` | step 9,961,472; success `0.857`, collision `0.143`, SNQI `0.100` | `d30f98f2c0a33515596043d54d87fa21d691c502eef255be8e43c702a0aef96f` | Conditional preserve only for ablation replay; below canonical leader. |
| `output/slurm/issue791-attention-head-job-12209/benchmarks/expert_policies/checkpoints/ppo_expert_issue_791_attention_head_promotion_10m_env22/ppo_expert_issue_791_attention_head_promotion_10m_env22_best.zip` | step 5,242,880; success `0.486`, collision `0.514`, SNQI `-0.996` | `33d4839bb9ed7c7e257f68a3bb874ee9d9a67a869f064df063c07ccb782e3015` | Do not promote; preserve only if the failed attention-head ablation needs exact replay. |

## Do Not Preserve As Training Artifacts

### Issue #857 Horizon-Matched PPO

Worktree: `../robot_sf_ll7.worktrees/857-ppo-horizon-matched`

The horizon100 policy is an archived negative ablation, not a promotion candidate. The issue note
explicitly rejects it after camera-ready evaluation collapsed to `success_mean=0.1489` with high
timeout share. Keep the context note and compact campaign summaries; no model publication is
recommended.

For traceability, the best training-eval checkpoint hash is:

```text
c2a94b9e8c7cb3e916a8fd9564e8321179b20a627763484f847995d18fb75087
output/slurm/issue791-reward-curriculum-job-12178/benchmarks/expert_policies/checkpoints/ppo_expert_issue_791_reward_curriculum_promotion_10m_env22_horizon100/ppo_expert_issue_791_reward_curriculum_promotion_10m_env22_horizon100_best.zip
```

### Predictive, Benchmark, And Workflow Worktrees

These worktrees contain benchmark evidence, negative predictive comparisons, workflow outputs, or
small smoke artifacts rather than training outputs requiring model preservation:

* `../robot_sf_ll7.worktrees/1427-predictive-slurm-runs`: negative obstacle-feature comparison;
  preserve compact summaries already used by the PR/issue, not model checkpoints.
* `../robot_sf_ll7.worktrees/1353-broader-amv-row-contract`: current #1353 benchmark campaign
  outputs and model-cache copies of already registered inputs; not new training.
* `../robot_sf_ll7.worktrees/1391-1392-slurm-workflow`: #1344/#1354 benchmark campaign summaries;
  not new training.
* `../robot_sf_ll7.worktrees/1398-snqi-rollup`: metric rollup evidence; not training.
* `../robot_sf_ll7.worktrees/1191-evaluate-ml-intern`: assistant/workflow smoke outputs; not
  training.
* `../robot_sf_ll7.worktrees/qwen-issue-1309-codex-smoke`: no relevant Robot SF training output.
* `../robot_sf_ll7.worktrees/origin-codex-best-learning-policy`: no inspected new training
  checkpoint that beats or replaces the registered policy.

## Cleanup Guardrails

Before deleting or pruning any worktree output:

1. Preserve or publish the #1108 job `12472` dataset and BC checkpoint if #1108 remains active.
2. Decide whether #1024 needs exact checkpoint replay; if yes, publish the env30 best checkpoint
   and optionally the env22 comparison checkpoint.
3. Treat older #791 policy-search checkpoints as superseded unless a specific ablation issue still
   needs exact replay.
4. Keep compact benchmark/report evidence under `docs/context/` or `docs/context/evidence/`, not
   raw `output/` bundles.
5. Re-check `squeue --me` and `sacct` before cleanup so running jobs are not disrupted.

## Dirty Worktree Caution

At audit time, several worktrees had local modifications or branch drift. Do not clean them with
destructive Git commands:

* `1108-bc-warm-start-ppo`: branch ahead/behind; contains the active #1108 artifacts above.
* `origin-2026-04-28-policy_search`: modified SLURM scripts.
* `origin-codex-best-learning-policy`: modified context note.
* `qwen-issue-1309-codex-smoke`: modified docs and far behind main.
