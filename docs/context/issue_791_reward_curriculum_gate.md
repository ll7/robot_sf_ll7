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

## Promotion Result (2026-04-30)

Job **12189** completed a 10M-step reward-curriculum promotion run from this worktree:

- SLURM log: `output/slurm/12189-issue791-reward-curriculum.out`
- Run summary:
  `output/slurm/issue791-reward-curriculum-job-12189/benchmarks/ppo_imitation/runs/ppo_expert_issue_791_reward_curriculum_promotion_10m_env22_20260429T195426.json`
- Expert manifest:
  `output/slurm/issue791-reward-curriculum-job-12189/benchmarks/expert_policies/ppo_expert_issue_791_reward_curriculum_promotion_10m_env22.json`
- W&B run: `ppo_expert_issue_791_reward_curriculum_promotion_10m_env22_20260429T195426`

Result:

- `success_rate.mean=0.3871428571428571`
- `collision_rate.mean=0.5021428571428571`
- `snqi.mean=-1.2549230908819666`
- Best checkpoint selected by success: `success_rate=0.5000` at `5,767,168` steps with
  `meets_convergence=False`

Interpretation: this is valid completed 10M evidence, but it is weak promotion evidence. The result
does not beat the earlier eval-aligned large-capacity issue-791 leader and should be treated as a
negative/underwhelming reward-curriculum-only promotion result. Use it to justify continuing with
the eval-aligned asymmetric-critic or attention-head follow-ups rather than promoting this policy.
