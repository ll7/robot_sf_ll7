# Open Issues Training Split Audit

Date: 2026-05-30

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1686>

## Scope

This note is a lightweight pointer for learned-policy artifact manifests. It does not run training,
move checkpoints, upload artifacts, or make benchmark claims. It identifies current training lanes
whose future checkpoints, normalizers, datasets, or adapter configs should cite the learned-policy
manifest fields in
[artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md).

## Training Lane Pointers

| Lane | Current source | Split/provenance implication |
| --- | --- | --- |
| learned risk model v1 | `configs/training/learned_risk_model_issue_1395_launch_packet.yaml` and [issue_1395_learned_risk_launch_packet.md](issue_1395_learned_risk_launch_packet.md) | Manifest must keep hard guards authoritative, preserve trace-contract checksums, and replace `pending` durable aliases before benchmark eligibility. |
| shielded PPO repair | `configs/training/shielded_ppo_issue_1396_launch_packet.yaml` and [issue_1396_shielded_ppo_launch_packet.md](issue_1396_shielded_ppo_launch_packet.md) | Manifest must state the PPO checkpoint lineage, repair config, comparison-freeze checksum, and guard/fallback policy. |
| oracle imitation dataset | `configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml` and [issue_1397_oracle_imitation_launch_packet.md](issue_1397_oracle_imitation_launch_packet.md) | Manifest must cite the dataset split contract, durable dataset artifacts, and whether the resulting policy is training-only/oracle or deployable. |
| ORCA residual BC | `configs/training/orca_residual/orca_residual_bc_issue_1428.yaml` and [issue_1428_orca_residual_lineage.md](issue_1428_orca_residual_lineage.md) | Manifest must separate base ORCA behavior, residual bounds, candidate config, diagnostic checksums, and guarded fallback behavior. |
| LiDAR PPO MLP | `configs/training/lidar/lidar_learned_policy_launch_packet_issue_1615.yaml` and [issue_1615_lidar_learned_policy_plan.md](issue_1615_lidar_learned_policy_plan.md) | Manifest must state the LiDAR observation schema, learned normalizer fit split if any, and cross-observation benchmark eligibility boundary. |

## Fail-Closed Rule

If a lane lacks a durable artifact URI, checksum, training config, training commit, observation
schema, action schema, normalizer URI or `not_required` declaration, license/access note, split
contract, or benchmark eligibility verdict, it may remain a launch packet or research-only record
but must not be promoted as a learned-policy benchmark row.

Worktree-local `output/` paths may describe exploratory runs or hydration targets only. They are not
durable learned-policy artifacts.

## Validation

For updates to this note:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```
