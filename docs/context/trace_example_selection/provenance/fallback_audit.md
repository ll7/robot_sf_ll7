# Fallback and mode audit (execution commit 373dbfde4f39667cf9e8732dabe7df5118bdeab1)

From the raw rows (each claim's `screening_ledger.csv`, `mode*` columns): every
screened (seed, arm) row of social_force (4), orca (4) and goal (4) has
`algorithm_metadata.status = ok`, `pedestrian_model.fallback_degraded_status = native`,
`integrity.effective_view.degraded = false`, so `mode = native` in all 12 rows. No arm's
row records a numeric fallback counter (`mode_fallback_counters = none_recorded`). The
absence of a counter does not prove zero fallback, so the source and frozen settings were
audited for what they allow:

- **orca.** The release planner entry sets `socnav_missing_prereq_policy: fail-fast`
  (`configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml:108`).
  Under fail-fast the runner forces `allow_fallback = False`
  (`robot_sf/benchmark/map_runner/map_runner.py:801-802`). `plan_velocity_world` takes the
  heuristic path only when `_ensure_rvo2()` returns False
  (`robot_sf/planner/socnav_orca.py:1334-1336`), and `_ensure_rvo2` returns True when
  rvo2 is importable and raises when it is not and fallback is off
  (`robot_sf/planner/socnav_orca.py:161-181`). rvo2 was importable in the run
  environment, so every orca step used the native RVO2 path
  (`_rvo2_velocity_world`, line 1419). The Python linear-program projection described as
  "ORCA fallback" (`socnav_orca.py:306-314`) is called only from `_solve_orca_velocity`
  inside the heuristic path (`socnav_orca.py:1583`), which was unreachable here.
  `planner_kinematics.execution_mode = adapter` is the release's differential-drive
  projection of the ORCA velocity, the same mode as in the release rows; it is not a
  fallback.
- **social_force.** `robot_sf/planner/socnav_social_force.py` contains no fallback path
  (no occurrence of "fallback"). `execution_mode = adapter` as in the release rows.
- **goal.** `execution_mode = native`; the goal baseline has no alternative path.

Recorded fallback count: 0 of 12 rows under the row markers above; the orca heuristic
fallback is ruled out by configuration plus environment, not by a counter.
