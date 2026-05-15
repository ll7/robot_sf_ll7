# Issue #1209 Imitation Observation Contract

Related issue: `ll7/robot_sf_ll7#1209`

## Goal

Issue #1209 fixes the observation-contract mismatch that blocked the #1108 BC warm-start PPO
experiment. The promoted BR-06 source checkpoint
`ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200` was trained with SocNav structured
observations plus occupancy-grid `env_overrides`, while the imitation scripts previously rebuilt
default `drive_state` / `rays` environments for trajectory collection, BC pretraining, and PPO
fine-tuning.

## Current Contract

The imitation path now treats the expert training config and trajectory metadata as the durable
source for environment reconstruction:

* collection accepts `--training-config`, applies its `env_overrides`, adapts observations to the
  loaded policy observation space, and stores the resulting observation contract in dataset
  metadata;
* BC pretraining rebuilds the environment from explicit config fields or dataset metadata and
  filters the env to the persisted observation keys before flattening;
* PPO fine-tuning uses `dataset_id` plus the same metadata/config path to rebuild both training and
  evaluation envs.

The reusable helper surface is `scripts/training/imitation_env_contract.py`. The issue #749 launch
packet configs now declare the BR-06 source training and scenario configs:

* `configs/training/ppo_imitation/bc_pretrain_issue_749_v10_warm_start.yaml`
* `configs/training/ppo_imitation/ppo_finetune_issue_749_v10_warm_start.yaml`

## Validation Path

Targeted regression tests:

```bash
uv run pytest tests/integration/test_ppo_pretraining_pipeline.py::test_issue_749_warm_start_configs_define_env_contract tests/integration/test_ppo_pretraining_pipeline.py::test_collect_trajectories_filters_to_policy_observation_space -q
```

One-episode non-dry-run collection against the real BR-06 checkpoint:

```bash
uv run python scripts/training/collect_expert_trajectories.py \
  --dataset-id issue_1209_preflight_b60iopxt_contract \
  --policy-id ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200 \
  --episodes 1 \
  --scenario-config configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml \
  --training-config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml \
  --seeds 111
```

Tiny BC smoke from that dataset:

```bash
uv run --group imitation python scripts/training/pretrain_from_expert.py \
  --config output/runs/issue_1209/bc_preflight.yaml
```

Tiny PPO fine-tune smoke from the BC checkpoint:

```bash
uv run python scripts/training/train_ppo_with_pretrained_policy.py \
  --config output/runs/issue_1209/ppo_finetune_preflight.yaml
```

The `output/runs/issue_1209/*.yaml` files are ignored local proof configs. The generated
trajectory dataset, BC checkpoint, fine-tune checkpoint, and run manifests under `output/` are
disposable validation artifacts unless a future #1108 execution promotes them to a durable store.

Observed local validation on 2026-05-14:

* targeted regression file: `6 passed`;
* non-dry-run collection: wrote `issue_1209_preflight_b60iopxt_contract` with `episodes=1`,
  `dry_run=false`, and a 20-key `Dict` observation contract;
* BC smoke: wrote `issue_1209_bc_preflight_policy.zip`;
* PPO fine-tune smoke: wrote `issue_1209_ppo_finetune_preflight_finetuned.zip`;
* launch-packet dry-runs: collection, BC pretraining, and PPO fine-tuning completed without
  observation-key mismatches;
* full PR readiness: `3481 passed, 21 skipped`.

## Follow-Up Boundary

#1209 only proves the launch path can replay the BR-06 checkpoint, train BC from the matching
dataset contract, and start PPO fine-tuning from the BC checkpoint. It does not complete the full
#1108 10M-step experiment or make a benchmark-strengthening claim.
