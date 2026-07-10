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
