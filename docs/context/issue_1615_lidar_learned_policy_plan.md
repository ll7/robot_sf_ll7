# Issue #1615 LiDAR Learned-Policy Launch Plan

Date: 2026-05-29

## Scope

This note records the launch packet for LiDAR-based learned local policies requested by issue
number 1615. It plans candidate baselines, feature-extractor variants, registry metadata, smoke
validation, and artifact promotion. It does not launch training, submit SLURM jobs, promote a
checkpoint, or claim benchmark performance.

## Launch Packet

- Packet:
  `configs/training/lidar/lidar_learned_policy_launch_packet_issue_1615.yaml`
- Eligibility specs:
  `configs/training/lidar/lidar_ppo_mlp_eligibility_issue_1615.yaml`
  and `configs/training/lidar/lidar_perception_adapter_eligibility_issue_1615.yaml`
- Observation source of truth: `docs/dev/observation_contract.md`
- Feature extractor overview: `docs/feature_extractors/README.md`
- Checklist helper: `scripts/validation/check_learned_policy_eligibility.py`

## Observation Boundary

The planned policies use `ObservationMode.DEFAULT_GYM` and benchmark observation level `lidar_2d`.
Deployment-time inputs are limited to `drive_state` and `rays`. Occupancy grids, SocNav structured
pedestrian state, future trajectories, simulator collision labels, and route outcome labels are
forbidden at evaluation time.

This matters because the future policy should test whether LiDAR-style perception is enough for a
local learned baseline, not whether privileged simulator state can be smuggled into a learned
planner.

## Candidate Set

- `ppo_lidar_mlp_gate_v1`: simple PPO baseline using the existing feature-extractor sweep base and
  an MLP extractor.
- `ppo_lidar_attention_gate_v1`: richer PPO perception variant using the attention extractor over
  ray/drive-state features.
- `ppo_lidar_lstm_history_v1`: history-oriented PPO variant using the existing LSTM extractor path.
- `dreamerv3_lidar_world_model_gate_v1`: research-only DreamerV3 candidate based on
  `configs/training/rllib_dreamerv3/drive_state_rays.yaml`.

The launch packet keeps these as future training candidates. A dedicated follow-up should
materialize fixed training configs and run only after the metadata preflight remains green.

## Registry Metadata

A future promoted checkpoint needs explicit registry fields for `observation_level`,
`observation_mode`, `observation_keys`, `ray_count`, `stack_steps`, `goal_encoding`,
`feature_extractor`, `action_contract`, `training_config`, `checkpoint_artifact_uri`,
`evaluation_config`, and `no_privileged_runtime_inputs`.

Until a durable checkpoint exists, the checklist verdict is `eligible_for_research_only`, not a
benchmark-ready adapter claim.

## Validation

Checklist preflight:

```bash
uv run python scripts/validation/check_learned_policy_eligibility.py \
  configs/training/lidar/lidar_ppo_mlp_eligibility_issue_1615.yaml \
  configs/training/lidar/lidar_perception_adapter_eligibility_issue_1615.yaml
```

Targeted tests:

```bash
uv run pytest -q \
  tests/training/test_lidar_learned_policy_launch_packet.py \
  tests/validation/test_check_learned_policy_eligibility.py \
  tests/test_feature_extractors.py
```

Expected result: both eligibility specs pass, feature extractors still import and instantiate, and
the launch packet test confirms LiDAR-only runtime inputs plus artifact-promotion boundaries.

## Follow-Up Boundary

Future training work should create dedicated issues for a PPO MLP smoke/short run, a richer
attention or LSTM perception variant, and eventual registry promotion. Any DreamerV3 or other
world-model LiDAR branch is further gated by the #1623 world-model feasibility decision: first open
a source/provenance and data-contract preflight, not a training smoke. Each follow-up must record
exact commit, config, seeds, ray count, stack steps, checkpoint artifact URI, smoke result, and
whether any guard or projection was active. Checkpoints and raw logs stay out of git; only compact
manifests or summaries should be tracked.
