# Issue #4016 Distributional RL Smoke Evidence

This directory records a diagnostic-only issue #4016 smoke comparison from one
QR-DQN-style checkpoint evaluated with both mean-return and lower-tail CVaR action
selection.

## Claim Boundary

- Evidence tier: `diagnostic-only`.
- Claim boundary: risk-selection diagnostic only; not benchmark evidence.
- Benchmark safety claim: false.
- No full benchmark campaign, Slurm/GPU job, or paper-facing claim was run or edited.

## Commands

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/training/train_distributional_rl.py \
  --config configs/training/distributional_rl/qr_dqn_issue_4016_smoke.yaml \
  --log-level WARNING

scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/analysis/materialize_distributional_rl_issue_4016_smoke_manifests.py \
  --training-manifest output/models/distributional_rl/issue_4016/training_manifest.json \
  --output-dir docs/context/evidence/issue_4016_distributional_rl_smoke \
  --write-comparison-config docs/context/evidence/issue_4016_distributional_rl_smoke/compare_config.yaml

scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/analysis/compare_distributional_rl_issue_4016.py \
  --config docs/context/evidence/issue_4016_distributional_rl_smoke/compare_config.yaml
```

## Result

- `distributional_rl_risk_comparison.json`: `comparison_status=valid_diagnostic`.
- Matched context: same checkpoint, seed, and total timesteps.
- Fallback/degraded rows: excluded `0`, included as non-evidence `0`.
- Smoke trainer result: `train_steps=112`; bulky checkpoint and trace remain under ignored
  `output/models/distributional_rl/issue_4016/`.

This is the smallest remaining empirical-output slice after PR #4476. It records real
mean/CVaR smoke manifests and leaves broader benchmark-strength evidence out of scope.
