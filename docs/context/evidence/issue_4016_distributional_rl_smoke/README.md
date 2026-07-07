# Issue #4016 Distributional RL Measured Smoke Evidence

This directory records diagnostic-only issue #4016 evidence from one QR-DQN-style checkpoint
evaluated by the benchmark runner in both mean-return and lower-tail CVaR action-selection modes.

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

scripts/dev/run_worktree_shared_venv.sh -- uv run robot_sf_bench --quiet baseline \
  --matrix configs/scenarios/sets/classic_cross_trap_subset.yaml \
  --algo distributional_rl \
  --algo-config configs/baselines/distributional_rl_issue_4016_mean.yaml \
  --benchmark-profile experimental \
  --out output/benchmarks/issue4016/mean/baseline_stats.json \
  --jsonl output/benchmarks/issue4016/mean/episodes.jsonl \
  --base-seed 4016 --repeats 1 --horizon 5 --dt 0.1 --workers 1 --no-resume

scripts/dev/run_worktree_shared_venv.sh -- uv run robot_sf_bench --quiet baseline \
  --matrix configs/scenarios/sets/classic_cross_trap_subset.yaml \
  --algo distributional_rl \
  --algo-config configs/baselines/distributional_rl_issue_4016_cvar.yaml \
  --benchmark-profile experimental \
  --out output/benchmarks/issue4016/cvar/baseline_stats.json \
  --jsonl output/benchmarks/issue4016/cvar/episodes.jsonl \
  --base-seed 4016 --repeats 1 --horizon 5 --dt 0.1 --workers 1 --no-resume

scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/analysis/summarize_distributional_rl_issue_4016_benchmark_rows.py \
  --mean-jsonl output/benchmarks/issue4016/mean/episodes.jsonl \
  --risk-jsonl output/benchmarks/issue4016/cvar/episodes.jsonl \
  --output-dir docs/context/evidence/issue_4016_distributional_rl_smoke

scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/analysis/audit_issue_4016_acceptance.py \
  --evidence-dir docs/context/evidence/issue_4016_distributional_rl_smoke
```

## Result

- `summary.json`: `benchmark_runner_measured=true`, included rows `mean=6`, `risk=6`.
- `distributional_rl_risk_comparison.json`: `comparison_status=valid_diagnostic`.
- `acceptance_audit.json`: `closure_status=complete`.
- Fallback/degraded rows: `0` excluded from both mean and risk inputs.

The ignored benchmark JSONL and checkpoint artifacts stay under `output/`; the tracked files here are
small reviewable summaries of the measured diagnostic smoke slice.
