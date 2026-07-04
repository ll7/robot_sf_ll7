# Issue #4017 Constrained RL Diagnostic Comparison

This directory documents the issue #4017 diagnostic-reporting slice. Generated smoke manifests,
checkpoints, traces, comparison reports, and readiness reports stay under ignored `output/`.

Claim boundary: comparison evidence is diagnostic-only. It records matched smoke manifest fields,
constraint budgets, budget violations, multiplier trajectory, runtime when present in the manifest,
and fallback/degraded status. It is not a benchmark-strength, paper-grade, or dissertation safety
claim.

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

If smoke manifests are missing, stale, degraded, or lack required fields, the report builder fails
closed and emits `diagnostic_blocked` with explicit blockers instead of treating the comparison as
evidence.

Consolidated readiness handoff:

```bash
uv run python scripts/analysis/check_constrained_rl_readiness_issue_4017.py \
  --report output/benchmarks/issue4017/comparison_report.json \
  --output output/benchmarks/issue4017/readiness_report.json
```

The readiness report is the integration surface for remaining issue #4017 work. It records whether
the diagnostic comparison is ready for the next paired empirical campaign, which blockers remain,
and the next empirical action. It keeps `ready_for_benchmark_claim` false until a separate
benchmark-strength campaign is run and reviewed.
