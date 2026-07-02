# Issue #4012 Offline-to-Online RL Fine-Tuning

Claim boundary: diagnostic implementation lane only; not benchmark evidence and not paper-facing.
Evidence status: smoke/diagnostic-only until a predeclared benchmark campaign runs. Major caveats:
fallback, degraded, or off-contract offline rows are fail-closed exclusions; worktree-local datasets
and checkpoints are not durable evidence. Uncertainty: high for performance effects; this slice only
proves workflow initialization and wiring.

## What Landed

- `RLTrajectoryDataset.v1` train-split rows can be loaded into a SAC offline transition batch.
- Observation and action rows are checked against the online environment spaces before replay use.
- SAC config has an opt-in `offline_online` block; disabled configs keep existing behavior.
- A diagnostic orchestrator runs an offline-online SAC arm and a from-scratch SAC arm with matching
  seed and timestep budget.

## Commands

Generate or provide a compatible `RLTrajectoryDataset.v1` JSONL and manifest under
`output/issue_4012_offline_online_rl_smoke/`, then run:

```bash
uv run --extra training python scripts/training/run_offline_online_rl.py \
  --config configs/training/offline_online_rl/issue_4012_sac_smoke.yaml
```

## Artifact Disposition

The default config writes checkpoints and summaries under `output/`, which is worktree-local and
ignored. Those files are diagnostic smoke artifacts only unless promoted through a durable artifact
store with provenance.

## Known Limitations

No full benchmark campaign, Slurm/GPU submission, Decision Transformer policy, or paper-facing
claim is included in this slice.

## Next Empirical Action

Run the smoke with a small compatible dataset, inspect the diagnostic summary, then decide whether a
durable benchmark campaign is justified.
