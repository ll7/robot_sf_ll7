# Policy Search Portfolio Overview

Generated: `2026-05-05T11:12:47.683544+00:00`

This note summarizes the registered policy-search candidates and the latest tracked reports.
It is an evidence map, not a paper-facing benchmark claim. Stages are not equally strong:
`full_matrix` evidence is stronger than `stress_slice`, which is stronger than
`nominal_sanity` and `smoke`.

## Current Leaders

| Candidate | Family | Evidence Stage | Decision | Episodes | Success | Collision | Near Miss | Report |
|---|---|---|---|---:|---:|---:|---:|---|
| scenario_adaptive_hybrid_orca_v1 | scenario_adaptive_classical | full_matrix | tracked | 141 | 0.9291 | 0.0213 | 0.4113 | `docs/context/policy_search/reports/2026-05-02_scenario_adaptive_hybrid_orca_v1_full_matrix.md` |
| hybrid_rule_v3_progress_2p4 | hybrid_rule_based | full_matrix | tracked | 144 | 0.2708 | 0.0139 | 0.3403 | `docs/context/policy_search/reports/2026-05-05_hybrid_rule_v3_progress_2p4_full_matrix.md` |
| hybrid_rule_v3_fast_progress_static_escape | hybrid_rule_based | full_matrix | tracked | 144 | 0.2639 | 0.0139 | 0.3403 | `docs/context/policy_search/reports/2026-05-05_hybrid_rule_v3_fast_progress_static_escape_full_matrix.md` |
| hybrid_rule_v0_minimal | hybrid_rule_based | full_matrix | tracked | 144 | 0.2569 | 0.0139 | 0.3264 | `docs/context/policy_search/reports/2026-05-05_hybrid_rule_v0_minimal_full_matrix.md` |
| hybrid_rule_v3_fast_progress | hybrid_rule_based | full_matrix | tracked | 144 | 0.2569 | 0.0139 | 0.3333 | `docs/context/policy_search/reports/2026-05-05_hybrid_rule_v3_fast_progress_full_matrix.md` |
| hybrid_rule_v3_dynamic_relaxed | hybrid_rule_based | full_matrix | tracked | 144 | 0.2431 | 0.0139 | 0.3125 | `docs/context/policy_search/reports/2026-05-05_hybrid_rule_v3_dynamic_relaxed_full_matrix.md` |
| hybrid_rule_v3_teb_like_rollout | hybrid_rule_based | full_matrix | tracked | 144 | 0.2431 | 0.0139 | 0.3194 | `docs/context/policy_search/reports/2026-05-05_hybrid_rule_v3_teb_like_rollout_full_matrix.md` |
| hybrid_rule_v3_waypoint2_dynamic_clearance | hybrid_rule_based | full_matrix | tracked | 144 | 0.2153 | 0.1042 | 0.3264 | `docs/context/policy_search/reports/2026-05-05_hybrid_rule_v3_waypoint2_dynamic_clearance_full_matrix.md` |

## Why The Best Candidates Look Good

- `scenario_adaptive_hybrid_orca_v1`: very high route-completion rate, low collision rate, low near-miss exposure, scenario-specific overrides are visible in the evidence; design idea: The retained fast-progress static-recenter hybrid candidate is strongest on most current full-matrix scenarios, while tuned ORCA cleanly resolves the remaining Francis leave-group hard-radius deadlock. A scenario-explicit classical selector should recover that case without changing the hybrid behavior on the other scenarios. Main remaining failure mode: `timeout_low_progress` (6).
- `hybrid_rule_v3_progress_2p4`: limited current success, low collision rate, low near-miss exposure; design idea: A moderate 2.4 m/s speed envelope with stronger progress pressure should recover long-route nominal-sanity timeouts without the near-miss and doorway regressions observed at 3.0 m/s. Main remaining failure mode: `timeout_low_progress` (63).
- `hybrid_rule_v3_fast_progress_static_escape`: limited current success, low collision rate, low near-miss exposure, scenario-specific overrides are visible in the evidence; design idea: Remaining fast-progress failures include static-clearance stalls where forward rollout is rejected but a short rotation can reopen a safe slow-forward heading. Reusing the slow static-escape gate, adding a scored static-recenter probe, and using a faster scenario-specific static-corridor creep should recover doorway, crossing, and long corridor stalls while preserving the hard static-collision gate. Main remaining failure mode: `timeout_low_progress` (64).
- `hybrid_rule_v0_minimal`: limited current success, low collision rate, low near-miss exposure; design idea: A clean deterministic DWA-style hybrid-rule control variant with explicit safety filtering and score diagnostics should provide a transparent non-learning baseline for later social, ORCA, recovery, and ensemble mechanisms. Main remaining failure mode: `timeout_low_progress` (65).
- `hybrid_rule_v3_fast_progress`: limited current success, low collision rate, low near-miss exposure; design idea: Raising the manually specified speed envelope to the 3.0 m/s robot limit used elsewhere in the repository should reduce low-progress timeouts on long route slices while preserving the v3 static and dynamic safety filters. Main remaining failure mode: `timeout_low_progress` (64).

## All Registered Candidates

| Candidate | Status | Family | Required Stages | Best Stage | Success | Collision | Analysis |
|---|---|---|---|---|---:|---:|---|
| scenario_adaptive_hybrid_orca_v1 | implemented | scenario_adaptive_classical | smoke, nominal_sanity, stress_slice, full_matrix | full_matrix | 0.9291 | 0.0213 | very high route-completion rate, low collision rate, low near-miss exposure, scenario-specific overrides are visible in the evidence; design idea: The retained fast-progress static-recenter hybrid candidate is strongest on most current full-matrix scenarios, while tuned ORCA cleanly resolves the remaining Francis leave-group hard-radius deadlock. A scenario-explicit classical selector should recover that case without changing the hybrid behavior on the other scenarios. Main remaining failure mode: `timeout_low_progress` (6). |
| hybrid_rule_v3_progress_2p4 | implemented | hybrid_rule_based | smoke, nominal_sanity, stress_slice | full_matrix | 0.2708 | 0.0139 | limited current success, low collision rate, low near-miss exposure; design idea: A moderate 2.4 m/s speed envelope with stronger progress pressure should recover long-route nominal-sanity timeouts without the near-miss and doorway regressions observed at 3.0 m/s. Main remaining failure mode: `timeout_low_progress` (63). |
| hybrid_rule_v3_fast_progress_static_escape | implemented | hybrid_rule_based | smoke, nominal_sanity, stress_slice | full_matrix | 0.2639 | 0.0139 | limited current success, low collision rate, low near-miss exposure, scenario-specific overrides are visible in the evidence; design idea: Remaining fast-progress failures include static-clearance stalls where forward rollout is rejected but a short rotation can reopen a safe slow-forward heading. Reusing the slow static-escape gate, adding a scored static-recenter probe, and using a faster scenario-specific static-corridor creep should recover doorway, crossing, and long corridor stalls while preserving the hard static-collision gate. Main remaining failure mode: `timeout_low_progress` (64). |
| hybrid_rule_v0_minimal | implemented | hybrid_rule_based | smoke, nominal_sanity, stress_slice | full_matrix | 0.2569 | 0.0139 | limited current success, low collision rate, low near-miss exposure; design idea: A clean deterministic DWA-style hybrid-rule control variant with explicit safety filtering and score diagnostics should provide a transparent non-learning baseline for later social, ORCA, recovery, and ensemble mechanisms. Main remaining failure mode: `timeout_low_progress` (65). |
| hybrid_rule_v3_fast_progress | implemented | hybrid_rule_based | smoke, nominal_sanity, stress_slice | full_matrix | 0.2569 | 0.0139 | limited current success, low collision rate, low near-miss exposure; design idea: Raising the manually specified speed envelope to the 3.0 m/s robot limit used elsewhere in the repository should reduce low-progress timeouts on long route slices while preserving the v3 static and dynamic safety filters. Main remaining failure mode: `timeout_low_progress` (64). |
| hybrid_rule_v3_dynamic_relaxed | implemented | hybrid_rule_based | smoke, nominal_sanity, stress_slice | full_matrix | 0.2431 | 0.0139 | limited current success, low collision rate, low near-miss exposure; design idea: Shortening only the hard dynamic collision horizon should reduce freezing in doorway and crossing scenes while retaining hard radius-based dynamic collision rejection, static footprint clearance, and route-guided progress. Main remaining failure mode: `timeout_low_progress` (68). |
| hybrid_rule_v3_teb_like_rollout | implemented | hybrid_rule_based | smoke, nominal_sanity, stress_slice | full_matrix | 0.2431 | 0.0139 | limited current success, low collision rate, low near-miss exposure; design idea: Adding an occupancy-grid route-guide candidate to the safety-filtered DWA scorer should recover progress in static/corridor local minima without giving back the collision improvement from the static footprint filter. Main remaining failure mode: `timeout_low_progress` (67). |
| hybrid_rule_v3_waypoint2_dynamic_clearance | implemented | hybrid_rule_based | smoke, nominal_sanity | full_matrix | 0.2153 | 0.1042 | limited current success, collision rate still needs work, low near-miss exposure; design idea: Slightly stronger dynamic-clearance scoring may reduce intrusive near misses without the static-collision regression caused by speed-cap comfort variants. Main remaining failure mode: `timeout_low_progress` (63). |
| hybrid_rule_v3_static_margin0 | implemented | hybrid_rule_based | smoke, nominal_sanity, stress_slice | full_matrix | 0.2153 | 0.1111 | limited current success, collision rate still needs work, low near-miss exposure; design idea: Enforcing static hard clearance over the full rollout while using no extra static margin beyond the reported robot radius should preserve static safety without the doorway freezing caused by a 5 cm buffer in tight passages. Main remaining failure mode: `timeout_low_progress` (62). |
| hybrid_rule_v3_static_margin0_waypoint2 | implemented | hybrid_rule_based | smoke, nominal_sanity | full_matrix | 0.2153 | 0.1111 | limited current success, collision rate still needs work, low near-miss exposure; design idea: Switching from the active route waypoint at 2.0 m instead of 0.9 m may reduce route-local stalls in crossing and corridor scenes while keeping the selected v3 static safety settings. Main remaining failure mode: `timeout_low_progress` (62). |
| hybrid_rule_v3_waypoint2_mild_comfort | implemented | hybrid_rule_based | smoke, nominal_sanity | full_matrix | 0.2014 | 0.0903 | limited current success, collision rate still needs work, low near-miss exposure; design idea: A mild dynamic-clearance increase on top of the waypoint2 candidate may reduce intrusive near misses without the static collision regression seen in the stronger comfort variant. Main remaining failure mode: `timeout_low_progress` (68). |
| hybrid_rule_v3_static_margin0_comfort | implemented | hybrid_rule_based | smoke, nominal_sanity | full_matrix | 0.1806 | 0.0972 | limited current success, collision rate still needs work, low near-miss exposure; design idea: Increasing dynamic clearance pressure and applying a lower speed cap out to 2.5 m should reduce doorway and crowd intrusive near misses without altering static safety. Main remaining failure mode: `timeout_low_progress` (75). |
| hybrid_rule_v3_static_margin0_waypoint3 | implemented | hybrid_rule_based | smoke, nominal_sanity | full_matrix | 0.1458 | 0.1111 | limited current success, collision rate still needs work, low near-miss exposure; design idea: A 3.0 m waypoint handoff distance may further reduce route-local stalls in long crossing/corridor scenes, but may be brittle around doorway waypoints. Main remaining failure mode: `timeout_low_progress` (67). |
| hybrid_orca_sampler_v1 | implemented | hybrid_model_based | smoke, nominal_sanity, stress_slice, full_matrix_if_promising | full_matrix | 0.0139 | 0.1111 | limited current success, collision rate still needs work, low near-miss exposure; design idea: Keep ORCA-like safety behavior while allowing a short-horizon sampler to recover progress in constrained geometry. Main remaining failure mode: `timeout_low_progress` (104). |
| hybrid_rule_v3_waypoint2_route_lookahead8_static05 | implemented | hybrid_rule_based | smoke, nominal_sanity | stress_slice | 0.2917 | 0.0000 | limited current success, low collision rate, low near-miss exposure; design idea: The 8-cell route lookahead may become acceptable if paired with a 5 cm static hard margin, trading some doorway progress for collision-free behavior. Main remaining failure mode: `timeout_low_progress` (11). |
| hybrid_rule_v3_waypoint2_speed2p2 | implemented | hybrid_rule_based | smoke, nominal_sanity | stress_slice | 0.2500 | 0.0833 | limited current success, collision rate still needs work, low near-miss exposure; design idea: A mild 2.2 m/s speed envelope on top of the selected waypoint2 policy may recover long-route timeout cases without the safety regressions seen in the more aggressive 2.4 m/s progress-weight retune. Main remaining failure mode: `timeout_low_progress` (10). |
| hybrid_rule_v3_waypoint2_route_lookahead8 | implemented | hybrid_rule_based | smoke, nominal_sanity | nominal_sanity | 0.3333 | 0.0556 | moderate progress relative to weak baselines, bounded collision rate, low near-miss exposure; design idea: Looking farther along the route-guide grid path may reduce long-route stalls in crossing, corridor, and Francis-style scenarios without changing safety margins or social comfort terms. Main remaining failure mode: `timeout_low_progress` (10). |
| hybrid_rule_v3_waypoint2_progress | implemented | hybrid_rule_based | smoke, nominal_sanity | nominal_sanity | 0.2778 | 0.0556 | limited current success, bounded collision rate, low near-miss exposure; design idea: Stronger manual progress and speed preference on top of waypoint2 may convert low-progress timeouts while preserving the same hard safety filters. Main remaining failure mode: `timeout_low_progress` (12). |
| hybrid_rule_v3_waypoint2_static_escape | implemented | hybrid_rule_based | smoke, nominal_sanity | nominal_sanity | 0.2778 | 0.0556 | limited current success, bounded collision rate, low near-miss exposure; design idea: Allowing only slow, non-worsening commands when the robot is already inside the conservative occupancy-grid static clearance band may reduce static deadlock timeouts without weakening occupied-cell collision rejection. Main remaining failure mode: `timeout_low_progress` (11). |
| hybrid_rule_v3_waypoint2_route_lookahead8_inflation4 | implemented | hybrid_rule_based | smoke, nominal_sanity | nominal_sanity | 0.2778 | 0.1111 | limited current success, collision rate still needs work, low near-miss exposure; design idea: The 8-cell route lookahead improved nominal success but caused one static collision; increasing only route-guide obstacle inflation may preserve the progress gain while steering away from the static hazard. Main remaining failure mode: `timeout_low_progress` (10). |
| hybrid_rule_v3_waypoint2_route_lookahead6 | implemented | hybrid_rule_based | smoke, nominal_sanity | nominal_sanity | 0.2222 | 0.0000 | limited current success, low collision rate, low near-miss exposure; design idea: A smaller route-guide lookahead increase may retain the nominal success gain suggested by 8 cells while avoiding its static-collision regression. Main remaining failure mode: `timeout_low_progress` (12). |
| hybrid_rule_v3_waypoint2_route_lookahead8_static02 | implemented | hybrid_rule_based | smoke, nominal_sanity | nominal_sanity | 0.2222 | 0.0000 | limited current success, low collision rate, low near-miss exposure; design idea: A smaller 2 cm static hard margin may remove the 8-cell route-lookahead collision while retaining more of its success gain than the 5 cm margin. Main remaining failure mode: `timeout_low_progress` (12). |
| hybrid_rule_v4_recovery_aware | implemented | hybrid_rule_based | smoke, nominal_sanity, stress_slice | nominal_sanity | 0.2222 | 0.0000 | limited current success, low collision rate, low near-miss exposure; design idea: Adding a narrow static-deadlock recovery mode to the route-guided hybrid planner should convert some safe low-progress stalls into goal progress without weakening dynamic-agent fail-closed behavior. Main remaining failure mode: `timeout_low_progress` (12). |
| hybrid_rule_v3_waypoint2_route_commit | implemented | hybrid_rule_based | smoke, nominal_sanity | nominal_sanity | 0.1667 | 0.1111 | limited current success, collision rate still needs work, low near-miss exposure; design idea: Giving the route-guide candidate a small deterministic bonus only after weak 3 s progress may reduce local route stalls while leaving normal DWA behavior unchanged in progressing scenes. Main remaining failure mode: `timeout_low_progress` (11). |
| hybrid_rule_v3_waypoint2_route_lookahead8_clearance1 | implemented | hybrid_rule_based | smoke, nominal_sanity | nominal_sanity | 0.1667 | 0.1111 | limited current success, collision rate still needs work, low near-miss exposure; design idea: Stronger route-guide clearance penalty may keep the 8-cell lookahead progress gain while avoiding the static collision without globally tightening the hard static margin. Main remaining failure mode: `timeout_low_progress` (11). |
| mpc_clearance_sampler_v1 | implemented | model_predictive_control | smoke, nominal_sanity, stress_slice | nominal_sanity | 0.1667 | 0.2778 | limited current success, collision rate still needs work, low near-miss exposure; design idea: A deterministic NMPC-style rollout scorer should improve constrained-geometry progress without giving up clearance control. Main remaining failure mode: `timeout_low_progress` (10). |
| risk_guarded_ppo_v1 | implemented | guarded_learning_policy | smoke, nominal_sanity, stress_slice | nominal_sanity | 0.1667 | 0.2778 | limited current success, collision rate still needs work, low near-miss exposure; design idea: The existing PPO success signal can be preserved if unsafe short-horizon actions are vetoed and replaced with safer local controls. Main remaining failure mode: `timeout_low_progress` (10). |
| scenario_adaptive_orca_v1 | implemented | adaptive_classical | smoke, nominal_sanity, stress_slice | nominal_sanity | 0.1667 | 0.0000 | limited current success, low collision rate, low near-miss exposure, scenario-specific overrides are visible in the evidence; design idea: Family-specific ORCA parameterization should reduce classic bottleneck risk without slowing Francis-style flowing interactions too aggressively. Main remaining failure mode: `timeout_low_progress` (13). |
| planner_selector_v1 | implemented | adaptive_ensemble | smoke, nominal_sanity, stress_slice | smoke | 0.0000 | 0.0000 | limited current success, low collision rate, low near-miss exposure; design idea: Existing non-learning and predictive heads can be combined into a stronger selector without introducing any new training dependency. Main remaining failure mode: `timeout_low_progress` (3). |
| learned_risk_model_v1 | slurm_handoff_required | learned_auxiliary_cost | n/a | not_run | n/a | n/a | A lightweight learned risk estimator should improve candidate ranking only when attached to a hard safety guard |

## Coverage Gaps

Candidates with no tracked report:
- `learned_risk_model_v1` (learned_auxiliary_cost)

Candidates that still need `full_matrix` evidence before broad comparison:
- `hybrid_rule_v3_waypoint2_route_lookahead8_static05`: best current stage `stress_slice` with success `0.2917` and collision `0.0000`
- `hybrid_rule_v3_waypoint2_speed2p2`: best current stage `stress_slice` with success `0.2500` and collision `0.0833`
- `hybrid_rule_v3_waypoint2_route_lookahead8`: best current stage `nominal_sanity` with success `0.3333` and collision `0.0556`
- `hybrid_rule_v3_waypoint2_progress`: best current stage `nominal_sanity` with success `0.2778` and collision `0.0556`
- `hybrid_rule_v3_waypoint2_static_escape`: best current stage `nominal_sanity` with success `0.2778` and collision `0.0556`
- `hybrid_rule_v3_waypoint2_route_lookahead8_inflation4`: best current stage `nominal_sanity` with success `0.2778` and collision `0.1111`
- `hybrid_rule_v3_waypoint2_route_lookahead6`: best current stage `nominal_sanity` with success `0.2222` and collision `0.0000`
- `hybrid_rule_v3_waypoint2_route_lookahead8_static02`: best current stage `nominal_sanity` with success `0.2222` and collision `0.0000`
- `hybrid_rule_v4_recovery_aware`: best current stage `nominal_sanity` with success `0.2222` and collision `0.0000`
- `hybrid_rule_v3_waypoint2_route_commit`: best current stage `nominal_sanity` with success `0.1667` and collision `0.1111`
- `hybrid_rule_v3_waypoint2_route_lookahead8_clearance1`: best current stage `nominal_sanity` with success `0.1667` and collision `0.1111`
- `mpc_clearance_sampler_v1`: best current stage `nominal_sanity` with success `0.1667` and collision `0.2778`
- `risk_guarded_ppo_v1`: best current stage `nominal_sanity` with success `0.1667` and collision `0.2778`
- `scenario_adaptive_orca_v1`: best current stage `nominal_sanity` with success `0.1667` and collision `0.0000`
- `planner_selector_v1`: best current stage `smoke` with success `0.0000` and collision `0.0000`

## Reproduction Commands

List candidates registered for a stage:

```bash
uv run python scripts/tools/summarize_policy_search_portfolio.py \
  --list-candidates --stage full_matrix
```

Submit a SLURM array for a stage:

```bash
scripts/dev/sbatch_policy_search_sweep.sh --stage full_matrix --dry-run
scripts/dev/sbatch_policy_search_sweep.sh --stage full_matrix --throttle 2
```

Refresh this overview after new reports land:

```bash
uv run python scripts/tools/summarize_policy_search_portfolio.py \
  --output-md docs/context/policy_search/portfolio_overview_2026-05-05.md
```
