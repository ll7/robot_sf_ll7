# Issue #4017 constrained-RL diagnostic comparison

This directory documents the small issue #4017 diagnostic-reporting slice. The
tracked artifact is the report contract and command path; generated smoke
manifests, checkpoints, traces, and comparison reports stay under ignored
`output/`.

Claim boundary: the comparison is diagnostic-only. It records matched smoke
manifest fields, constraint budgets, budget violations, multiplier trajectory,
runtime when present in the manifest, and fallback or degraded status. It does
not make a benchmark-strength, paper-grade, or dissertation safety claim.

Canonical commands:

```bash
uv run python scripts/training/train_constrained_rl.py \
  --config configs/training/ppo/issue_4017_unconstrained_smoke.yaml \
  --log-level WARNING

uv run python scripts/training/train_constrained_rl.py \
  --config configs/training/ppo/issue_4017_constrained_smoke.yaml \
  --log-level WARNING

uv run python scripts/analysis/compare_constrained_rl_issue_4017.py \
  --config configs/benchmarks/issue_4017_constrained_rl_diagnostic.yaml
```

Expected generated outputs:

- `output/benchmarks/issue4017/comparison_report.json`
- `output/benchmarks/issue4017/comparison_report.md`

If smoke manifests are missing, stale, degraded, or lack required fields, the
report builder fails closed or emits `diagnostic_blocked` with explicit blockers
instead of treating the comparison as evidence.
