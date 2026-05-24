# Shielded PPO Repair Campaign

## Goal

Test whether the existing PPO success signal can be improved further by
fine-tuning or retraining under stronger safety-aware reward or curriculum
constraints, while keeping the guarded runtime wrapper.

## Starting Points

- `configs/policy_search/candidates/risk_guarded_ppo_v1.yaml`
- `configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml`
- guarded PPO benchmark configs under `configs/algos/`

## Launch Packet

Issue #1396 adds the pre-SLURM launch packet:

- `configs/training/shielded_ppo_issue_1396_launch_packet.yaml`
- `scripts/validation/validate_shielded_ppo_launch_packet.py`
- `docs/context/issue_1396_shielded_ppo_launch_packet.md`

The packet encodes one repair hypothesis only: keep the BR-06 v3 PPO architecture and runtime
guard unchanged while increasing the route-completion collision penalty for fine-tuning. A
follow-up SLURM issue must validate the packet, materialize durable baseline/checkpoint artifacts,
and preserve the smoke/nominal-sanity stop gates before stress or full-matrix escalation.

## Handoff Rules

- keep the runtime guard active in all evaluations,
- record exact training config and checkpoint provenance,
- compare against the frozen PPO baseline and `risk_guarded_ppo_v1`.
