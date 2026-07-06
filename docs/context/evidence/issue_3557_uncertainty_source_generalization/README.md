# Issue #3557 Uncertainty-Source Generalization

Plain-language summary: this artifact runs the controlled #3471 episode harness across registered ScenarioBelief uncertainty sources. It records per-source oracle/retained/dropped aggregate decisions without promoting the diagnostic result into benchmark evidence.

- Issue: #3557
- Schema: `uncertainty_source_generalization_report.v1`
- Evidence tier: `diagnostic`
- Generalization verdict: `generalizes`
- Seeds: `[101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]`
- Command: `uv run python scripts/validation/run_uncertainty_source_generalization_issue_3557.py --config configs/benchmarks/scenario_belief_episode_safety_issue_3471.yaml`

## Claim Boundary

Controlled #3471 crossing scenario using real stream_gap planner and ScenarioBelief uncertainty gate. This is diagnostic-tier uncertainty-source generalization evidence only: no full benchmark campaign, no Slurm/GPU submission, no paper/dissertation claim.

This is not a full benchmark campaign result. It does not use Slurm or GPU resources and does not edit paper or dissertation claims.

## Per-Source Decisions

| Source | Condition builder | Decision | Unsafe-rate delta | Min-separation delta (m) |
| --- | --- | --- | ---: | ---: |
| existence_degradation | _condition_existence_degraded | reproduces_unsafe_drop | 16.75 | -0.4518 |
| visibility_occlusion | _condition_visibility_limited | inconclusive | 0.0 | 0.0 |
| covariance_inflation | _condition_covariance_inflated | reproduces_unsafe_drop | 16.75 | -0.4518 |
| class_probability | _condition_class_probability | reproduces_unsafe_drop | 16.75 | -0.4518 |
| tracking_noise | _condition_tracking_noise | reproduces_unsafe_drop | 18.333333 | -0.0713 |

Detailed machine-readable outputs:

- `summary.json`
- `per_source_decisions.csv`
