# Issue #3207 Fidelity Sensitivity Actual Slice 2026-06-20

- Status: `actual_campaign_slice`
- Evidence classification: `bounded_actual_slice`
- Git head: `3b272897a`
- Git worktree dirty at generation: `True`
- Raw rows: `output/fidelity_sensitivity/issue_3207_actual_slice_2026-06-20/episode_rows.jsonl`
- Claim boundary: bounded_actual_campaign_slice_not_full_benchmark_evidence: executes real Robot SF episodes for a compact two-planner local fidelity-sensitivity slice. It measures internal sensitivity on this slice only; it is not simulator-realism, sim-to-real, paper-facing planner-ranking, or full #3207 acceptance evidence.

## Scope

- Scenario set: `configs/scenarios/sets/paper_cross_kinematics_v1.yaml`
- Episodes: `54`
- Horizon: `180`
- Seeds: `111, 112, 113`
- Planners: `baseline_social_force, goal_seek`
- Limitation: local torch/rvo2-independent slice uses two non-learned planners; full config fixed_scope planners remain future work.
- Result caveats: `ranking_stability_is_on_bounded_two_planner_slice_only, full_fixed_scope_planners_not_run, all_observed_success_rates_zero, all_observed_collision_rates_one`

## Rank Stability

- Nominal ranking: `baseline_social_force, goal_seek`
- Rank stable on this slice: `True`
- Flipping variants: `none`

| Variant | Kendall tau | Rank flips | Top-1 changed |
|---|---:|---:|---|
| `integration_timestep__dt_0_05` | 1 | 0 | `False` |
| `integration_timestep__dt_0_20` | 1 | 0 | `False` |
| `social_force_speed_archetypes__mixed_balanced` | 1 | 0 | `False` |
| `social_force_speed_archetypes__rush_hour` | 1 | 0 | `False` |
| `observation_noise__pose_heading_low` | 1 | 0 | `False` |
| `observation_noise__pedestrian_dropout_low` | 1 | 0 | `False` |
| `clearance_radius__radius_0_30` | 1 | 0 | `False` |
| `clearance_radius__radius_0_50` | 1 | 0 | `False` |

This evidence measures internal simulator-fidelity sensitivity for the bounded local slice only.
It must not be cited as simulator-realism, sim-to-real, full benchmark, or paper-facing ranking evidence.
