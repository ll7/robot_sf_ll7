# Issue #3557 Uncertainty-Representation Generalization

Plain-language summary: this artifact runs the merged #4187 controlled episode harness across `belief_drop`, `conformal_radius`, and `envelope_inflation` representations. It asks whether dropping uncertain agents remains worse in the same diagnostic crossing scenario.

- Issue: #3557
- Schema: `uncertainty_representation_generalization_report.v1`
- Evidence tier: `diagnostic`
- Generalization verdict: `generalizes`
- Seeds: `[101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]`
- Command: `uv run python scripts/validation/run_uncertainty_representation_generalization_issue_3557.py --config configs/benchmarks/scenario_belief_episode_safety_issue_3471.yaml`

## Claim Boundary

Controlled #3471 crossing scenario using the real stream_gap planner and ScenarioBelief uncertainty gate. This is diagnostic-tier representation generalization evidence only: no full benchmark campaign, no Slurm/GPU submission, and no paper/dissertation claim edit.

This is not a full benchmark campaign result. It does not use Slurm or GPU resources and does not edit paper or dissertation claims.

## Per-Representation Decisions

| Representation | Harness decision | Generalization verdict | Unsafe-rate delta | Min-separation delta (m) |
| --- | --- | --- | ---: | ---: |
| belief_drop | revise | reproduces_unsafe_dropping | 0.246424 | -0.4518 |
| conformal_radius | revise | reproduces_unsafe_dropping | 0.246424 | -0.4518 |
| envelope_inflation | revise | reproduces_unsafe_dropping | 0.246424 | -0.4518 |

Detailed machine-readable outputs:

- `summary.json`
- `per_representation_decisions.csv`
- `integration_report.md`
