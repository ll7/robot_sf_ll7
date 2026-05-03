# Shielded PPO Repair Campaign

## Goal

Test whether the existing PPO success signal can be improved further by
fine-tuning or retraining under stronger safety-aware reward or curriculum
constraints, while keeping the guarded runtime wrapper.

## Starting Points

- `configs/policy_search/candidates/risk_guarded_ppo_v1.yaml`
- `configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml`
- guarded PPO benchmark configs under `configs/algos/`

## Handoff Rules

- keep the runtime guard active in all evaluations,
- record exact training config and checkpoint provenance,
- compare against the frozen PPO baseline and `risk_guarded_ppo_v1`.