# Issue 791 Reward Curriculum Gate

## Goal

Implement the first bounded PPO improvement from issue 791: a config-first reward curriculum that
switches between staged `route_completion_v3` reward settings after completed episodes.

The asymmetric critic and attention-head ideas remain follow-up work and are not part of this
slice.

## What Changed

- `robot_sf/gym_env/reward.py`
  - Added `build_reward_curriculum_function(...)`.
  - Added a stateful curriculum wrapper that advances at terminal episode boundaries.
- `robot_sf/gym_env/environment_factory.py`
  - Added `reward_curriculum` support to `make_robot_env(...)` and `make_image_robot_env(...)`.
- `scripts/training/train_ppo.py`
  - Startup logs now report curriculum-based reward profiles explicitly.
- Tests
  - Added unit coverage for reward curriculum advancement.
  - Added factory and training-contract coverage for curriculum wiring.
- Config
  - Added `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_stage1.yaml`.

## Canonical Config

- `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_stage1.yaml`

## Validation

```bash
uv run pytest tests/gym_env/test_reward_registry.py -q
uv run pytest tests/training/test_train_expert_ppo_contract.py -q
uv run pytest tests/integration/test_train_expert_ppo.py -q
```

All three targeted suites passed.

## Follow-Up

- Keep issue 791 scoped to the first reward-curriculum gate unless the next slice is explicitly
  approved.
- Treat attention-head and asymmetric-critic work as separate issues.
