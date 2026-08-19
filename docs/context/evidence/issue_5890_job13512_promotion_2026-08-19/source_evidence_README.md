# Issue #3207 Fidelity Sensitivity Actual Slice 2026-07-16

- Status: `actual_campaign_slice`
- Evidence classification: `bounded_actual_slice`
- Git head: `c153848d7`
- Git worktree dirty at generation: `False`
- Raw rows: `ignored_output/slurm/02b-issue3207-fidelity-full-fixed-scope-job-13512/campaign/raw/episode_rows.jsonl`
- Claim boundary: bounded_actual_campaign_slice_not_full_benchmark_evidence: executes real Robot SF episodes for a compact two-planner local fidelity-sensitivity slice. It measures internal sensitivity on this slice only; it is not simulator-realism, sim-to-real, paper-facing planner-ranking, or full #3207 acceptance evidence.

## Scope

- Scenario set: `configs/scenarios/classic_interactions_francis2023.yaml`
- Episodes: `5184`
- Horizon: `120`
- Seeds: `111, 112, 113`
- Planners: `baseline_social_force, goal_seek, orca`
- Limitation: local torch/rvo2-independent slice uses two non-learned planners; full config fixed_scope planners remain future work.
- Result caveats: `ranking_stability_is_on_bounded_two_planner_slice_only, full_fixed_scope_planners_not_run`

## Rank Stability

- Nominal ranking: `goal_seek, orca, baseline_social_force`
- Rank evidence status: `identifiable`
- Rank identifiability reason: `none`
- Rank stable on this slice: `True`
- Flipping variants: `none`
- Non-identifiable variants: `none`

| Variant | Rank evidence | Kendall tau | Rank flips | Top-1 changed |
|---|---|---:|---:|---|
| `integration_timestep__dt_0_05` | `identifiable` | 1 | 0 | `False` |
| `integration_timestep__dt_0_20` | `identifiable` | 1 | 0 | `False` |
| `social_force_speed_archetypes__mixed_balanced` | `identifiable` | 1 | 0 | `False` |
| `social_force_speed_archetypes__rush_hour` | `identifiable` | 1 | 0 | `False` |
| `observation_noise__pose_heading_low` | `identifiable` | 1 | 0 | `False` |
| `observation_noise__pedestrian_dropout_low` | `identifiable` | 1 | 0 | `False` |
| `clearance_radius__radius_0_30` | `identifiable` | 1 | 0 | `False` |
| `clearance_radius__radius_0_50` | `identifiable` | 1 | 0 | `False` |

This evidence measures internal simulator-fidelity sensitivity for the bounded local slice only.
It must not be cited as simulator-realism, sim-to-real, full benchmark, or paper-facing ranking evidence.
