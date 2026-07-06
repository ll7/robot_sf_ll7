# Issue #3215 Scenario-Family Oracle Packet

This is a CPU-generated launch packet for the next diagnostic step after the hard-case portfolio synthesis. It does not run paired seeds or establish benchmark evidence.

- Status: `ready`
- Evidence tier: `diagnostic-only`
- Packet: `configs/benchmarks/predictive_scenario_family_oracle_arm_issue_3215.yaml`
- Git head: `d768ab6c217263ef02917928cddd29b42315f3a1`
- Scenario family: `predictive_hardcase_forecast_risk_oracle_family_v1`
- Rows per seed: `324`
- Paired seeds: `30`
- Forecast arms: `4`
- Planned paired rows: `38880`

## Forecast Arms

| Arm | Role | Oracle state |
| --- | --- | --- |
| `none` | control | `false` |
| `constant_velocity` | deployable_baseline | `false` |
| `interaction_aware` | deployable_interaction_model | `false` |
| `oracle_future` | non_deployable_upper_bound | `true` |

## Checks

| Check | Status | Detail |
| --- | --- | --- |
| `schema_version` | PASS | predictive_scenario_family_oracle_packet.v1 |
| `issue` | PASS | 3215 |
| `evidence_tier` | PASS | diagnostic-only |
| `forecast_arms` | PASS | found=['constant_velocity', 'interaction_aware', 'none', 'oracle_future'] |
| `oracle_boundary` | PASS | Oracle rows are diagnostic upper bounds only. They must not be mixed with deployable rows as planner success evidence. |
| `outcomes` | PASS | found=['collision_rate', 'false_positive_stop_rate', 'forecast_risk_calibration', 'near_miss_rate', 'progress_loss', 'stop_timing'] |
| `paired_seed_count` | PASS | 30 |
| `scenario_family_factorization` | PASS | factor_sizes={'occlusion_geometries': 3, 'approach_angles_deg': 3, 'actor_speeds_mps': 3, 'time_to_arrival_offsets_s': 3, 'sensor_latency_ms': 2, 'uncertainty_levels': 2}, rows_per_seed=324 |
| `claim_boundary` | PASS | Launch packet only. This packet does not run paired seeds, establish a benchmark result, promote a planner, or support paper/dissertation claims. |
| `scenario_family.base_scenario_set` | PASS | configs/scenarios/sets/predictive_hardcase_portfolio_v1.yaml |
| `source_context.scenario_seed_manifest` | PASS | configs/benchmarks/predictive_hard_seeds_v1.yaml |
| `source_context.forecast_risk_gate` | PASS | configs/research/forecast_risk_coupling_issue_2916.yaml |
| `source_context.prerequisite_synthesis` | PASS | docs/context/evidence/issue_3215_hardcase_portfolio_synthesis_2026-06-21/README.md |

## Boundary

Launch packet only. This packet does not run paired seeds, establish a benchmark result, promote a planner, or support paper/dissertation claims.

Out of scope: full benchmark campaign run, Slurm or GPU submission, planner promotion, paper or dissertation claim edit.
