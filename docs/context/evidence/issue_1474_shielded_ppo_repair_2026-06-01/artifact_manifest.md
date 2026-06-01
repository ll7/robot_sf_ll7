# Issue #1474 Shielded PPO Repair Artifact Manifest 2026-06-01

Date: 2026-06-01

## Training Run

- Issue: #1474 / #1396 shielded-PPO repair campaign.
- SLURM job: `12674`, `COMPLETED`, exit `0:0`, `a30`, elapsed `07:06:33`.
- Training commit: `cc3e8552b0fa1ae47ddb3f42cd74443576c6e9c0`.
- Training config:
  `configs/training/ppo/ablations/expert_ppo_issue_1474_shielded_repair_collision20_5m.yaml`.
- Launch packet: `configs/training/shielded_ppo_issue_1396_launch_packet.yaml`.
- Local synced root: ignored worktree-local SLURM cache; not durable evidence.
- W&B run: `ll7/robot_sf/d8w8uykh`.
- W&B model artifact:
  `ll7/robot_sf/ppo_expert_issue_1474_shielded_repair_collision20_5m-best-success:v5`.

## Best Checkpoint

- Policy id: `ppo_expert_issue_1474_shielded_repair_collision20_5m`.
- Local synced checkpoint: ignored worktree-local SLURM cache; not durable evidence.
- Size: `166187385` bytes.
- SHA256: `7714123d79a4a75ba7e83df61b36cf3a5145191bd00e905e122e0bf87679cc26`.
- W&B aliases: `step-5000000`, `best-success`, `latest`.
- W&B digest: `e43f6c0245038faf763ed1cd5efd42f5`.

## Final Training Eval

The selected best-success checkpoint was the final 5M-step checkpoint:

| Metric | Value |
| --- | ---: |
| `success_rate` | `0.83` |
| `collision_rate` | `0.16` |
| `snqi` | `-0.10229038095238098` |
| `path_efficiency` | `0.8145328571428573` |
| `eval_episode_return` | `27.92607950528125` |
| `comfort_exposure` | `0.02583333333333333` |

## Checksums

| Artifact | SHA256 |
| --- | --- |
| `benchmarks/expert_policies/ppo_expert_issue_1474_shielded_repair_collision20_5m.json` | `7b58faf76e7fe4b8fc545ec6d7d54922af28b14db26d877a587191d93f3504c8` |
| `benchmarks/expert_policies/checkpoints/ppo_expert_issue_1474_shielded_repair_collision20_5m/ppo_expert_issue_1474_shielded_repair_collision20_5m_best.summary.json` | `5fd598c6d4ec1b33e2e6b1ac325765ab7d5124e380d9cf6cd43b0a571b86633a` |
| `benchmarks/ppo_imitation/eval_timeline/ppo_expert_issue_1474_shielded_repair_collision20_5m_20260601T042334.json` | `e275edef18043c88183b33322029fdaada71deb17f77d5873feb54d32666e52e` |

## Artifact Classification

- `durable-promoted`: W&B artifact
  `ll7/robot_sf/ppo_expert_issue_1474_shielded_repair_collision20_5m-best-success:v5`.
- `tracked-compact-evidence`: this manifest and
  `docs/context/evidence/issue_1474_shielded_ppo_repair_2026-06-01/learned_policy_eligibility.yaml`.
- `non-evidence-local-only`: synced raw SLURM logs and local ignored-cache copies.
- `ignored-cache`: local model cache if hydrated by the registry.

## Claim Boundary

This is training and artifact-provenance evidence only. The checkpoint is not benchmark evidence
and not a guarded-policy promotion until `shielded_ppo_issue1474_collision20_v1` passes the
runtime-guarded smoke and nominal-sanity stop gates with diagnostics preserved.

## Guarded Smoke Gate

SLURM job `12685_0` completed the first post-training guarded smoke on 2026-06-01. The generic
policy-search report says `pass`, but the launch-packet smoke gate fails because success was `0.0`
instead of the required `1.0`. Collision and near-miss rates were both `0.0`; the single episode
timed out with failure mode `overconservative_stop`.

The smoke JSONL reports `shield_decision_count=80`, `shield_intervention_count=0`,
`shield_override_count=0`, and `decision_counts.goal_reached=80`. This means the runtime wrapper
returned zero motion throughout the smoke episode rather than proving a useful guarded policy.
