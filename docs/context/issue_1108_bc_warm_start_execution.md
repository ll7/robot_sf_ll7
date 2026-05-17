# Issue #1108 BC Warm-Start PPO Execution

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1108>

## Goal

Run the #749 BC-preinitialized PPO experiment chain against the v10 warm-start contract:

1. collect expert trajectories from
   `ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200`,
2. train `issue_749_bc_preinit_v10_policy` with behavioral cloning,
3. fine-tune `issue_749_ppo_finetune_v10_warm_start`,
4. compare the final checkpoint against `27dbe5xu` and `b60iopxt`.

## Blocker Found

The 2026-05-14 preflight stopped in trajectory collection because the imitation pipeline did not
share a stable observation contract:

* the collector built default `drive_state`/`rays` envs while the b60iopxt checkpoint expects
  SocNav + occupancy-grid dict observations,
* applying the source env overrides exposed the next mismatch: current SocNav emits
  `robot_velocity_xy` and `robot_angular_velocity`, while the older checkpoint saved a 20-key dict
  observation space,
* BC and PPO fine-tune also built default envs, so a collector-only workaround would not make the
  chain valid.

## Unblock Implemented

The imitation pipeline now carries and applies env contract metadata:

* `configs/training/ppo_imitation/bc_pretrain_issue_749_v10_warm_start.yaml`
* `configs/training/ppo_imitation/ppo_finetune_issue_749_v10_warm_start.yaml`
* `scripts/training/collect_expert_trajectories.py`
* `scripts/training/imitation_env_contract.py`
* `scripts/training/pretrain_from_expert.py`
* `scripts/training/train_ppo_with_pretrained_policy.py`

Collection also resolves expert checkpoints through `model/registry.yaml` when the
`output/benchmarks/expert_policies/<policy>.zip` link is absent, and adapts live dict observations
to the loaded checkpoint's saved observation space before `policy.predict(...)`. The dataset still
stores the target v10 observation payload.

## Validation

Earlier lightweight checks on the Auxme login node passed before the 2026-05-17 merge with
`origin/main`. Those branch-local checks are superseded by the post-merge validation below.

Functional preflight:

```bash
LOGURU_LEVEL=WARNING uv run python scripts/training/collect_expert_trajectories.py \
  --dataset-id issue_749_b60iopxt_v10_eval_trajectories \
  --policy-id ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200 \
  --episodes 1 \
  --scenario-config configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml \
  --env-config configs/training/ppo_imitation/ppo_finetune_issue_749_v10_warm_start.yaml \
  --seeds 111
```

Result: passed. The local manifest reported `quality=validated`, `episode_count=1`,
`dry_run=false`, `env_contract_config=configs/training/ppo_imitation/ppo_finetune_issue_749_v10_warm_start.yaml`,
and `observation_mode=socnav_struct`.

Dry-run BC and PPO fine-tune also completed after the preflight dataset existed. Those dry-run
checkpoints are non-evidence and must not be promoted.

BC device smoke after the 2026-05-15 Slurm failures:

```bash
uv run --group imitation python - <<'PY'
# Constructed a tiny Gym env and one imitation Trajectory, then called
# scripts.training.pretrain_from_expert._create_bc_trainer(...).
PY
```

Result: passed after pinning both the SB3 PPO wrapper and the imitation BC trainer to CPU:
`ppo_device cpu`, `bc_policy_device cpu`.

Post-merge checks after syncing `issue-1108-bc-warm-start-ppo` with `origin/main` on
2026-05-17:

```bash
uv run python -m py_compile robot_sf/training/imitation_config.py \
  scripts/training/imitation_env_contract.py scripts/training/collect_expert_trajectories.py \
  scripts/training/pretrain_from_expert.py scripts/training/train_ppo_with_pretrained_policy.py
uv run ruff check robot_sf/training/imitation_config.py scripts/training/imitation_env_contract.py \
  scripts/training/collect_expert_trajectories.py scripts/training/pretrain_from_expert.py \
  scripts/training/train_ppo_with_pretrained_policy.py tests/training/test_imitation_env_contract.py \
  tests/training/test_pretrain_from_expert.py tests/training/test_train_ppo_with_pretrained_policy.py \
  tests/integration/test_ppo_pretraining_pipeline.py
uv run ruff format --check robot_sf/training/imitation_config.py \
  scripts/training/imitation_env_contract.py scripts/training/collect_expert_trajectories.py \
  scripts/training/pretrain_from_expert.py scripts/training/train_ppo_with_pretrained_policy.py \
  tests/training/test_imitation_env_contract.py tests/training/test_pretrain_from_expert.py \
  tests/training/test_train_ppo_with_pretrained_policy.py \
  tests/integration/test_ppo_pretraining_pipeline.py
uv run pytest tests/training/test_imitation_env_contract.py \
  tests/training/test_pretrain_from_expert.py \
  tests/training/test_train_ppo_with_pretrained_policy.py \
  tests/integration/test_ppo_pretraining_pipeline.py -q
bash -n SLURM/Auxme/issue_1108_bc_warm_start.sl
```

Result: passed (`22 passed` for the targeted pytest set). The merge resolution now uses
`scripts/training/imitation_env_contract.py` as the shared environment-contract boundary instead
of the deleted `robot_sf/training/env_overrides.py` helper.

## Slurm Submission

First attempt:

* job: `12461`
* result: failed immediately before training
* cause: wrapper used `uv sync --all-extras --group imitation`, which selects incompatible
  dependency sets (`rllib` conflicts with `imitation`)

Second attempt:

* job: `12462`
* partition/QoS: `a30` / `a30-gpu`
* node at startup: `auxme-imech172`
* result: failed in BC after collecting the full 141-episode dataset
* log: `output/slurm/12462-issue1108-bc-warm-start.out`
* result root after job exit:
  `output/slurm/issue1108-bcppo-job-12462/`
* cause: BC env still used the default scenario contract, producing flat observations of length
  `77163` while the collected dataset used the pinned scenario length `77095`

Third attempt:

* job: `12463`
* partition/QoS: `a30` / `a30-gpu`
* result: failed in BC after collecting the full 141-episode dataset
* log: `output/slurm/12463-issue1108-bc-warm-start.out`
* result root after job exit:
  `output/slurm/issue1108-bcppo-job-12463/`
* cause: imitation BC moved the policy to CUDA while demonstration action tensors remained on CPU

Fourth attempt:

* job: `12471`
* partition/QoS: `a30` / `a30-gpu`
* result: failed in BC after collecting the full 141-episode dataset
* log: `output/slurm/12471-issue1108-bc-warm-start.out`
* result root after job exit:
  `output/slurm/issue1108-bcppo-job-12471/`
* cause: setting `device="cpu"` on the SB3 PPO wrapper was insufficient because
  `imitation.algorithms.bc.BC` also has `device="auto"` and moved the supplied policy to CUDA

Fifth attempt:

* job: `12472`
* partition/QoS: `a30` / `a30-gpu`
* status checked at 2026-05-15 12:38 Europe/Berlin: running
* log: `output/slurm/12472-issue1108-bc-warm-start.out`
* evidence so far: collection completed, BC completed (`12472.1` completed with exit `0:0`), and
  PPO fine-tune launched (`12472.2` running; observed `total_timesteps=10240`, `success_rate=0.01`)
* result root after job exit:
  `output/slurm/issue1108-bcppo-job-12472/`

The checked-in launcher was patched after submission to use a job-local uv environment via
`UV_PROJECT_ENVIRONMENT=${WORKDIR}/uv_env` for future resubmits. Job `12462` had already started
with the previous launcher and temporarily adjusted the worktree `.venv`; this is local-only and
not part of the durable evidence.

## Follow-Up Boundary

Issue #1108 is not complete when job `12472` is merely running. Completion still requires:

* full 141-episode dataset manifest,
* real BC checkpoint and manifest,
* real PPO fine-tuned checkpoint and manifest,
* durable artifact pointers for dataset/checkpoints,
* policy-analysis comparison against `27dbe5xu` and `b60iopxt`,
* an outcome statement covering final score and sample efficiency.

Local files under `output/` remain cache/non-evidence until promoted through a durable pointer.
