# Issue #4014 RecurrentPPO LSTM Dry-Run Lane

This slice adds the CPU-safe smoke-prep lane for the true recurrent LSTM row in
issue #4014. It uses `sb3_contrib.RecurrentPPO` for non-dry-run training, while
`--dry-run` validates the config and writes a manifest without requiring the
optional package.

Claim boundary: this is dry-run and smoke-preparation evidence only. It does not
run a full benchmark campaign, submit Slurm or GPU work, compare PPO / PPO-LSTM /
PPO-Mamba results, or update paper-facing claims.

Canonical dry-run command:

```bash
uv run python scripts/training/train_recurrent_ppo.py \
  --config configs/training/ppo/issue_4014_ppo_lstm_recurrent_smoke.yaml \
  --dry-run \
  --log-level WARNING
```

The emitted `training_manifest.json` records `algorithm=recurrent_ppo`,
`policy=MultiInputLstmPolicy`, the LSTM policy kwargs, the optional dependency
install hint, and the dry-run claim boundary.
