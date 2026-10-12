# Canonical PPO Training Workflow

Use the config-driven PPO trainer for current expert-policy training:

```bash
uv run python scripts/training/train_ppo.py \
  --config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml
```

For a fast local pipeline check, add `--dry-run`:

```bash
uv run python scripts/training/train_ppo.py \
  --config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml \
  --dry-run \
  --log-level WARNING
```

`scripts/training/train_ppo.py` is the supported PPO expert training entrypoint. The legacy
`scripts/training_ppo.py` entrypoint fails closed with a migration command so old notes or shell
history cannot silently launch an unsupported run.

## Multi-Map Train/Test Protocol

Issue #4982 adds a declared geometry-held-out PPO protocol at
`configs/training/ppo/issue_4982_multi_map_train_test_protocol.yaml`. It keeps named training and
held-out scenario IDs disjoint, evaluates both splits at every checkpoint, and writes
`zero_shot_success_rate_decay` to the training manifest. Positive decay means the final held-out
success rate was lower than the train-split success rate. This metric is a run readout, not
generalization evidence until a predeclared campaign completes.

The same config declares training-only domain randomization for pedestrian speed multiplier,
route-spawn density multiplier, and route-spawn jitter. The sampled values are applied when a
training scenario environment is created; held-out evaluation remains unrandomized. The protocol
and randomization profile are recorded in the copied config, training-manifest notes, and optional
Weights & Biases run config.

For the environment boundary, reward ownership, benchmark-outcome distinction, and reviewer
run-record checklist, see
[Robot SF Environment Contract And Training Provenance](environment_contract.md).

## Evaluation Cadence

PPO training YAML files loaded through `load_expert_training_config()` must define
`evaluation.step_schedule`; this field controls checkpoint and evaluation cadence in training
timesteps. `evaluation.frequency_episodes` is deprecated and ignored when present. It may remain in
older YAML files for compatibility, but it does not control current checkpoint cadence.

`frequency_episodes` was episode-count based, which made cadence harder to compare across scenario
mixes, variable episode lengths, and worker-count changes. Use `evaluation.step_schedule` instead so
checkpointing and evaluation are tied to the same timestep surface that Stable-Baselines3 and the
training manifests report.

Implementation note: the `EvaluationSchedule` dataclass still keeps `frequency_episodes` as a
required constructor field and `step_schedule=()` as a default for compatibility with older helper
construction paths. The config loader enforces the current PPO training YAML contract and rejects
configs that rely on `frequency_episodes` alone.

## Startup Summary

At startup, the canonical trainer logs the resolved run-critical settings needed for provenance:
policy id, config path, scenario config, total timesteps, reward profile, requested and resolved
`num_envs`, worker mode, seed randomization, resume source, and scenario-sampling keys.

## Released-checkpoint warm start

For the released plain PPO in 0.0.8 (October B1002), prepare a new-run ZIP without training:

```bash
uv run python -m scripts.training.prepare_released_ppo_warm_start \
  --model-id ppo_release_robot_b1002_last_20261001 \
  --init-seed 100001 --out output/ftplan/init100001/model.zip
```

Pass that ZIP as `--resume-from` to the canonical trainer and retain the adjacent
provenance JSON with the run artifacts. Actor and value tensors are preserved;
timestep/update/episode counters and optimizer moments start fresh. A different
initialization seed changes later stochastic draws, not the released deterministic
policy. A prepared ZIP is a derived artifact, not a replacement registry release.

The released checkpoint has raw dict observations and no observation-normalization
companion. For a different checkpoint using `VecNormalize`, use
`robot_sf.training.released_warm_start.load_warm_start` with an environment and an
explicit pinned `normalization_path`; train with the returned effective environment
and save its normalization companion alongside every checkpoint. ZIP files do not
embed those observation statistics. The preparation CLI is restricted to the
known unnormalized release; it cannot be used to silently discard another source's
normalization. Observation/action spaces must match; action-contract or observation
schema migrations require a separately validated conversion.

B1002 uses velocity-delta actions with bounds `[-2, -1]` to `[2, 1]`, a 2.0 m/s
forward cap, no reverse, no curriculum, `route_completion_v3`, 18 workers and CNN
channels 32/64/64. The pilot preserves this interface, rather than using the April
issue-791 or March guarded-PPO source. Its benchmark action-contract receipt must
confirm unchanged velocity-delta semantics and bounds. Environment binding still
rejects any incompatible observation or action space.

The opt-in generic `target_action_space` API can change clipping bounds with
identical shape/dtype while preserving raw actor/value tensors. It does not convert
action semantics and is not used by the B1002 pilot.

To reproduce release parity without simulator stepping or GPU training, resolve
the registry-pinned B1002 release into the model cache, then run:

```bash
ROBOT_SF_B1002_PARITY_CHECKPOINT=output/model_cache/ppo_release_robot_b1002_last_20261001/ppo_release_robot_b1002_last_20261001-model.zip \
  scripts/dev/run_worktree_shared_venv.sh -- pytest -n0 -o addopts= \
  tests/training/test_released_warm_start.py -q
```

The release test checks its digest and serialized recipe, exact fixed-observation
actions, critic predictions and all actor/value tensors, and repeats parity after
saving/reloading each of the three initialization seeds. Without the explicit
checkpoint path this large-artifact test skips; the synthetic regressions still run.
