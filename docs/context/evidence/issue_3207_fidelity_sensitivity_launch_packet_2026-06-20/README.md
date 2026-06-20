# Issue #3207 Fidelity Sensitivity Launch Packet 2026-06-20

- Status: `launch_packet_only`
- Study: `issue_3207_fidelity_sensitivity_v1`
- Config: `configs/research/fidelity_sensitivity_v1.yaml`
- Git head: `c553e797c`
- Claim boundary: Protocol and metric contract only. This config defines deliberate simulation-fidelity sensitivity probes and the rank-stability/drift outputs required before benchmark claims can state a validity boundary. It is not benchmark evidence, sensor-realism evidence, sim-to-real evidence, or a paper-facing planner-ranking result until the sweep is run and promoted through a compact evidence bundle.


## Scope

- Scenario set: `configs/benchmarks/paper_experiment_matrix_v1.yaml`
- Seeds: `111, 112, 113`
- Planner groups: `orca, default_social_force, hybrid_rule_v0_minimal`
- Ranking metric: `snqi`

## Fidelity Axes

| Axis | Variants | Baseline Variant | Rationale |
|---|---:|---|---|
| `integration_timestep` | 3 | `dt_0_10_nominal` | Detect whether planner rankings depend on the simulator integration step. |
| `social_force_speed_archetypes` | 3 | `homogeneous_standard_nominal` | Detect whether pedestrian speed heterogeneity changes ranking stability. |
| `observation_noise` | 3 | `none_nominal` | Detect whether ranking stability depends on non-calibrated robustness noise. |
| `clearance_radius` | 3 | `radius_0_40_nominal` | Detect whether planner rankings are brittle to collision/proxemic radius assumptions. |

## Result Contract

Before this packet can support a validity-boundary claim, the sweep output must report:
- `axis`
- `variant`
- `planner`
- `seed`
- `scenario_id`
- `metric_values`
- `rank_metric`
- `kendall_tau_vs_baseline`
- `rank_flip_count`
- `per_metric_drift_vs_baseline`

Any axis with `rank_flip_count > 0` is a caveat/calibration candidate.

## Next Command Template

```bash
uv run python scripts/<fidelity_sweep>.py --config configs/research/fidelity_sensitivity_v1.yaml --out output/fidelity_sensitivity/
```

This packet is not benchmark evidence until a sweep is run and promoted with provenance.
