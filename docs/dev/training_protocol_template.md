# Training Protocol Template

Use this template to document training and evaluation runs. Copy into your experiment
notes or PR description and fill out each section.

## Summary
- Experiment name:
- Goal / hypothesis:
- Owner:
- Date range:

## Configuration
- Scenario file(s):
- Scenario split policy: `train` / `holdout`
- Split validation: `scripts/tools/split_scenarios.py --scenario <path>`
- Map selection: `map_id` or random from `map_pool`

## Observation Contract
- Observation mode: `default_gym` or `socnav_struct`
- Stacking (`stack_steps`):
- Occupancy grid: `use_occupancy_grid` / `include_grid_in_observation`
- Contract reference: `docs/dev/observation_contract.md`

## Reward Definition
- Reward function source: `robot_sf/gym_env/reward.py` or custom
- Shaping terms / weights:
- Any reward clipping or normalization:

## Force Flags
- Static obstacle forces: `peds_have_static_obstacle_forces`
- Ped-robot repulsion: `peds_have_robot_repulsion` or `sim_config.prf_config.is_active`
- Deprecated alias used: `peds_have_obstacle_forces` (yes/no)

## Seeds & Determinism
- Master seed:
- Training seed list:
- Evaluation seed list:
- Determinism settings (env vars, torch flags):

## Training Command
```bash
<command here>
```

## Evaluation Protocol
- Holdout scenarios:
- Evaluation cadence:
- Metrics and thresholds:

## Artifacts & Outputs
- Artifact root: `output/` or `ROBOT_SF_ARTIFACT_ROOT` override
- Checkpoints path:
- Episode logs path:
- Figures path:

## Validation
- Tests run:
- Smoke checks:
- Benchmark replay:

## Notes / Deviations
- Known limitations:
- Future follow-ups:
