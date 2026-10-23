# PPO release-contract retraining plan

Prepare four seeded replicas (1001/1002 per recipe), identical action interpretation,
release drive limits and observation contracts. No scheduler submission or registry promotion.
PR #9995 is the explicit code dependency; its commit is carried for executable validation.
The owned delta-training changes start after that dependency.

1. Add new leaves inheriting the exact A/B recipes; keep reward, network, curriculum,
   scenarios and 10M/15M budgets.
2. Prove the old acceleration-input training path fails parity against the release delta
   adapter; opt in to signed delta actions, sum before release clipping, convert to drive
   acceleration and integrate at 0.1 seconds.
3. Verify real environment transitions against release adapter and an independent oracle.
4. Run each leaf for 2048 CPU transitions with one environment, then prepare private
   admission packets, commit/push branches and open draft PRs.

Seeds are fixed for requested independent replicas; periodic evaluation uses dev seed 1003.
Outputs and smoke evidence are preserved outside the checkout. CPU smoke overrides device,
worker count and step count only; it is startup evidence, not checkpoint quality evidence.
The guard is deployment-only arbitration; parity concerns the base policy command and plant.
No performance, release admission or OOD claim is made. Orchestrator owns remote provisioning,
canonical queue admission and final submission.

## Declared interface and claim boundary

The four new training leaves use `env_overrides.ppo_action_semantics: velocity_delta`.
The four matching CPU evaluation profiles declare `action_semantics: velocity_delta`.
This follows #9995 and the historical drive update documented in
[ppo_checkpoint_action_semantics.md](ppo_checkpoint_action_semantics.md): a negative
linear delta brakes from current speed. Absolute targets would change that interpretation.

The signed policy output is added to the float32 physical speed exposed by SocNav,
then clipped to linear [0, 2] m/s and angular [-1, 1] rad/s. Both paths convert that
target to acceleration using dt=0.1 and the real drive clips acceleration and braking
at 1 m/s² and angular acceleration at 1 rad/s². Signed policy bounds are [-2, 2]
m/s per policy step and [-1, 1] rad/s per policy step; no dt scaling is applied to
the delta itself. Sensor keys, raw speed values, grid, extractor and per-arm foresight
are inherited unchanged. Native acceleration actions remain the default for other envs.

Guarded PPO retains deployment-only arbitration. The retrained base policy learns under
the same action interpretation and plant as its primary release proposal; the guard can
still deliberately replace a proposal. Training does not acquire a new shielding method.

These seeded replicas intentionally set `randomize_seeds: false`; the original leaves
ignore their listed seeds. This is required for two independently identifiable runs.
Periodic evaluation uses dev seed 1003, preserving episode counts, schedule and selection
metric. This is still in-distribution training on the existing scenario sets, with no
OOD or checkpoint performance claim.

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 scripts/dev/run_worktree_shared_venv.sh --profile training -- uv run pytest \
 tests/training/test_ppo_release_contract.py tests/baselines/test_ppo_action_semantics.py \
 tests/differential_drive_test.py tests/test_action_adapters.py \
 tests/gym_env/test_robot_env_snqi_step_metadata.py -q

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 scripts/dev/run_worktree_shared_venv.sh --profile training -- uv run python \
 -m scripts.validation.smoke_ppo_release_contract \
 --config configs/training/ppo/ablations/expert_ppo_release_contract_a_seed1001.yaml \
 --output /absolute/external/smoke-artifact-directory
```

Repeat the smoke command for each of the four leaves. It runs genuine SB3 learning
for 2048 transitions, including an optimizer update, using one environment on CPU.
Its output is startup evidence only. Full checkpoints must subsequently be evaluated
on dev seeds and registered with exact source, checksums and W&B coordinates before
the matching model_id-only evaluation profiles can load them. No existing registry
row is modified by the retraining changes. Slurm mechanics and admission packets are
maintained in the private overlay; the orchestrator performs admission and submission.

## Review fixes

All four leaves load the checked-in `ppo_release_contract_dev_eval_seeds.yaml` through
`evaluation.evaluation_seed_manifest`; the effective tuple is `(1003,)`. The loader
consumes each evaluation key and rejects any leftovers, including the schema-recognized
but unsupported inline `evaluation_seeds` field. Deprecated `frequency_episodes` is
explicitly consumed with its existing warning; checkpoint cadence remains `step_schedule`. Disabled legacy policy-analysis switches
are consumed explicitly; enabling these unimplemented features is rejected.
The regression checks the resolved tuple and per-episode seed selection, plus rejection
of the previously silently ignored key. Real transition tests compare nonzero float32
`robot_speed` observations against independent state values before and after stepping.
The inherited tracking recipe is unchanged; TensorBoard is required only when enabled.
