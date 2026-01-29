# Prompt: Analyze Latest Policy Sweep

You are a coding agent working in the Robot SF repo. Analyze the latest policy analysis sweep runs and produce a concise comparison report.

## Input runs (update as needed)
- `output/benchmarks/policy_analysis_fast_pysf_planner_20260128_132624`
- `output/benchmarks/policy_analysis_ppo_20260128_132624`
- `output/benchmarks/policy_analysis_socnav_orca_20260128_132624`

## Required outputs
- A markdown report under `specs/409-planner-comparison/` with:
  - Aggregate metrics table (success, collision rate, time_to_goal_norm, avg_speed, near_misses, comfort, jerk/curvature raw + filtered, energy).
  - Worst collision scenarios (top 5) per planner.
  - Lowest success scenarios (bottom 5) per planner.
  - Top problem episodes with video paths.
  - Diagnostics section (path_efficiency saturation, shortest_path_len presence, low-speed filter behavior).
  - If frames are extracted, include a Frame Snapshots section listing PNG paths.

## Analysis steps (must follow)
1) **Verify inputs exist**: each run directory has `episodes.jsonl` and `summary.json`.
2) **Aggregate metrics** from `episodes.jsonl`:
   - success_rate, collision_rate
   - mean: time_to_goal_norm, avg_speed, near_misses, comfort_exposure
   - mean: jerk_mean, curvature_mean
   - mean: jerk_mean_eps0p1, curvature_mean_eps0p1 (if present)
   - mean: energy, ped_force_q95, clearing_distance_min/avg
3) **Scenario extremes** from `summary.json` aggregates:
   - Top 5 collision scenarios (by collisions_mean)
   - Bottom 5 success scenarios (by success_mean)
4) **Problem episodes**: score = collisions*10 + comfort_exposure*2 + failure flag. List top 5 per run.
5) **Video check**: ensure each problem episode has a corresponding mp4 in `output/recordings/<run_name>/` using naming convention `<scenario>_seed<seed>_<policy>.mp4`.
6) **Diagnostics**:
   - Path-efficiency saturation: fraction of episodes with path_efficiency >= 0.999.
   - Shortest-path presence: check `shortest_path_len` exists in metrics for all episodes.
   - Low-speed filter: report low_speed_frac and explain why ε=0.1 is used.
7) **Optional frames**: extract 25/50/75% frames for top problem episodes using ffmpeg into `output/analysis/<report_id>/frames/`.

## Notes
- Use CET/CEST timestamps when creating new report names.
- Clearly label baseline categories: oracle/GT (fast_pysf), heuristic (socnav_orca), learned (ppo).
- If any required files are missing, explain what was found and what is missing.
