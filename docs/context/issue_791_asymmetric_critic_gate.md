# Issue 791 Asymmetric Critic Gate

## Goal

Implement the asymmetric-critic slice of issue 791: keep the shared SocNav + grid actor surface,
but add a critic-only privileged state vector built from the live observation payload plus
episode/state metadata.

## What Changed

- `robot_sf/gym_env/robot_env.py`
  - Added `asymmetric_critic` support to the observation space and runtime observation payload.
  - The privileged state is attached as `critic_privileged_state`.
- `robot_sf/training/ppo_policy.py`
  - Added `AsymmetricGridSocNavPolicy` for separate actor/critic feature extractors.
- `scripts/training/train_ppo.py`
  - Added config-driven policy selection and fail-closed validation for incompatible settings.
- `robot_sf/gym_env/environment_factory.py`
  - Threaded `asymmetric_critic` through the robot/image factory path.
- Tests
  - Added coverage for privileged-state extraction, env attachment, and policy selection.
- Config
  - Added `configs/training/ppo/ablations/expert_ppo_issue_791_asymmetric_critic_stage1.yaml`.

## Canonical Config

- `configs/training/ppo/ablations/expert_ppo_issue_791_asymmetric_critic_stage1.yaml`

## Validation

Run the targeted tests after the code changes are finalized:

```bash
uv run pytest tests/test_grid_socnav_extractor.py -q
uv run pytest tests/test_socnav_env_integration.py -q
uv run pytest tests/training/test_train_expert_ppo_contract.py -q
```

## Follow-Up

- Keep the asymmetric critic limited to the grid + SocNav policy path for now.
- Treat the attention-head slice as a separate follow-up change.
