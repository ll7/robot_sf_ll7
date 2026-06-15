# Issue #2837 Horizon and Timestep Ablation Report

## Scope

This analysis-only bundle compares forecast output horizon and output-timestep
presets for the prediction lane. It evaluates a constant-velocity Gaussian
baseline on durable repository trace fixtures across:

- Horizon ladder: 0.5 s, 1.0 s, 1.6 s, 2.0 s, 3.0 s
- Output dt_s ladder: 0.1 s, 0.2 s, 0.4 s, 0.5 s

## Evidence status

- `schema`: `HorizonTimestepAblation.v1`
- `claim_boundary`: `analysis_only_not_navigation_evidence`
- `reason`: the report varies only forecast output horizon and output timestep on
  open-loop trace fixtures. It does not change simulator physics step semantics,
  and long-horizon / coarse-dt rows are frequently unavailable because the
  durable fixtures are short (1-2 s).

## Files

- [ablation_report.json](ablation_report.json): full ablation rows, summary rows, and preset recommendations
- [ablation_report.md](ablation_report.md): human-readable Markdown report
- [summary.json](summary.json): compact issue coverage summary

## Reproducible command

```bash
uv run python scripts/benchmark/build_horizon_timestep_ablation_report.py \
  --issue 2837 \
  --generated-at-utc 2026-06-15T00:00:00+00:00 \
  --output-dir docs/context/evidence/issue_2837_horizon_timestep_ablation_2026-06-15
```

## Validation

```bash
uv run pytest tests/benchmark/test_horizon_timestep_ablation.py
uv run ruff check scripts/benchmark/build_horizon_timestep_ablation_report.py
uv run ruff format scripts/benchmark/build_horizon_timestep_ablation_report.py
```

## Claim boundary

The report is useful for choosing forecast-output presets for diagnostic
forecast evaluation. It does not prove navigation improvement, safety, human
realism, benchmark-strength predictor quality, or closed-loop planner benefit.
Long-horizon presets are unavailable in this fixture set because the durable
traces are too short.
