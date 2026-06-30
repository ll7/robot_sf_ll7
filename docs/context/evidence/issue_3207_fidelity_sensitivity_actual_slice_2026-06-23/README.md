# Issue #3207 Fidelity Sensitivity Actual Slice 2026-06-23

- Status: `actual_campaign_slice`
- Evidence classification: `bounded_actual_slice`
- Git head: `2ff5e07df`
- Git worktree dirty at generation: `False`
- Raw rows: `ignored_output/fidelity_sensitivity/issue_3207_actual_slice_2026-06-23/episode_rows.jsonl`
- Claim boundary: bounded_actual_campaign_slice_no_claim: executes real Robot SF episodes for a compact two-planner local fidelity-sensitivity slice. It measures internal sensitivity on this slice only; it is not benchmark evidence, not simulator-realism evidence, not sim-to-real evidence, and not paper-facing evidence.

## Scope

- Scenario set: `configs/scenarios/sets/paper_cross_kinematics_v1.yaml`
- Episodes: `30`
- Horizon: `120`
- Seeds: `111, 112, 113`
- Planners: `baseline_social_force, goal_seek`
- Limitation: local torch/rvo2-independent slice uses two non-learned planners; full config fixed_scope planners remain future work.
- Result caveats: `ranking_stability_is_on_bounded_two_planner_slice_only, full_fixed_scope_planners_not_run, rank_non_identifiable_primary_metric_zero_variance, all_observed_success_rates_zero, some_observed_collision_rates_one`

## Rank Stability

- Nominal deterministic order (ties broken by name): `baseline_social_force, goal_seek`
- Rank evidence status: `non-identifiable`
- Rank identifiability reason: `primary_metric_zero_variance`
- Rank stable on this slice: `not_applicable`
- Flipping variants: `none`
- Non-identifiable variants: `integration_timestep__dt_0_05, social_force_speed_archetypes__mixed_balanced, observation_noise__pose_heading_low, clearance_radius__radius_0_30`

| Variant | Rank evidence | Kendall tau | Rank flips | Top-1 changed |
|---|---|---:|---:|---|
| `integration_timestep__dt_0_05` | `non-identifiable: primary_metric_zero_variance` | NA | NA | NA |
| `social_force_speed_archetypes__mixed_balanced` | `non-identifiable: primary_metric_zero_variance` | NA | NA | NA |
| `observation_noise__pose_heading_low` | `non-identifiable: primary_metric_zero_variance` | NA | NA | NA |
| `clearance_radius__radius_0_30` | `non-identifiable: primary_metric_zero_variance` | NA | NA | NA |

This evidence measures internal simulator-fidelity sensitivity for the bounded local slice only.
It must not be cited as benchmark evidence, simulator-realism evidence, sim-to-real evidence, or paper-facing evidence.
