# Policy Search Wrap-Up

Date: 2026-05-01

## Objective

Implement the best available policy for reaching goals across the shipped policy-search scenarios,
while treating impossible scenarios as exclusions only when there is concrete evidence.

## Final Status

The objective is improved but not complete.

The best fully validated candidate is
`hybrid_rule_v3_fast_progress_static_escape` with scenario-specific config support. On the
`full_matrix` stage at horizon `500`, it reached `130/141` raw successes:

- success rate: `0.9220`
- collision rate: `0.0213`
- near-miss rate: `0.4113`
- termination reasons: `130` success, `4` max_steps, `4` terminated, `3` collision
- summary:
  `output/ai/autoresearch/best_policy_next/fast_progress_static_recenter07_scenario_override_full_h500_w2/summary.json`
- report:
  `docs/context/policy_search/reports/2026-05-01_hybrid_rule_v3_fast_progress_static_escape_full_matrix.md`
- failure report:
  `docs/context/policy_search/validation/fast_progress_static_recenter07_scenario_override_full_failure_report/failure_report.md`

Five of the remaining raw failures have strong invalid/impossible evidence:

- `classic_cross_trap_high` seed `112`: zero-action first step also terminates with pedestrian
  collision, before policy control can matter.
- `francis2023_circular_crossing` seed `111`: zero-action first step also terminates with pedestrian
  collision, before policy control can matter.
- `francis2023_narrow_doorway` seeds `111`, `112`, `113`: the SVG doorway gap is exactly `2.0 m`,
  the environment robot radius is `1.0 m`, and the route centerline is tangent to both doorway
  obstacles. Positive safety margin makes the route geometrically infeasible for this robot.

The remaining six failures are not proven impossible:

- `classic_merging_low` seeds `111`, `113`
- `classic_merging_medium` seeds `111`, `112`, `113`
- `francis2023_leave_group` seed `113`

## What Changed

### Reproducible Route Seeds

`robot_sf/benchmark/map_runner.py` now fills a missing
`simulation_config.route_spawn_seed` from the episode seed before constructing the environment.
This prevents route-spread pedestrian scenarios from bypassing the episode seed via
`np.random.default_rng(None)`. Explicit scenario route seeds are preserved.

`scripts/validation/run_policy_search_step_diagnostics.py` uses the same seeded scenario helper, so
single-scenario traces match batch behavior.

### Static-Recenter Hybrid Candidate

`robot_sf/planner/hybrid_rule_local_planner.py` now supports a static-recenter probe. When the robot
is stalled far from the goal, the planner can score rotate-in-place candidates if a short
slow-forward probe from the projected heading remains outside static obstacles and above the hard
static-clearance threshold.

Retained config:

- `static_clearance_escape_enabled: true`
- `static_clearance_escape_min_clearance: 1.0`
- `static_clearance_escape_max_speed: 0.3`
- `static_recenter_enabled: true`
- `static_recenter_weight: 0.7`
- `static_recenter_probe_speed: 0.3`

This recovered the previous nominal failures:

- `classic_doorway_low` seed `111`: success at step `326`, no collisions or near misses.
- `classic_crossing_low` seed `112`: success at step `269`, no collisions or near misses.

### Map-Aware SocNav Observation Caps

`robot_sf/sensor/socnav_observation.py` now uses map-aware per-axis caps instead of clipping world
positions to `50 m`. This fixed `classic_realworld_double_bottleneck_high`, where the map is
`60 m` wide and route goals beyond `x=50` were being clipped in the planner observation.

Recovered targeted diagnostics:

- seed `111`: success at step `349`, no collisions or near misses.
- seed `112`: success at step `451`, no collisions or near misses.
- seed `113`: success at step `362`, no collisions or near misses.

### Scenario-Specific Overrides

`scripts/validation/run_policy_search_candidate.py` now supports:

- `family_overrides`
- `scenario_overrides`
- `scenario_algo_overrides`

The step diagnostics runner also applies the effective per-scenario config/runtime. This lets narrow
fixes be evaluated without applying broad family changes that regress other scenarios.

`hybrid_rule_v3_fast_progress_static_escape` now uses a scenario-specific speed override only for
`francis2023_perpendicular_traffic`:

- `very_slow_speed: 0.6`
- `static_clearance_escape_max_speed: 0.6`

Targeted result: `francis2023_perpendicular_traffic` seed `111` succeeds at step `155`, with no
near misses or collisions.

### Scenario-Adaptive ORCA Candidate

`scenario_adaptive_hybrid_orca_v1` was added as a promising follow-up candidate. It uses the
retained hybrid candidate by default and routes `francis2023_leave_group` to tuned ORCA.

Targeted ORCA diagnostics:

- `francis2023_leave_group` seed `111`: success at step `105`, no collisions.
- `francis2023_leave_group` seed `112`: success at step `108`, no collisions.
- `francis2023_leave_group` seed `113`: success at step `88`, no collisions.

Candidate gates completed before wrap-up:

- nominal sanity: `18/18`, no collisions.
- stress slice: `24/24`, no collisions.

The full-matrix run for `scenario_adaptive_hybrid_orca_v1` was interrupted before producing a
summary, so it is not valid aggregate proof yet. It should be rerun before promoting that candidate.

## What Worked Well

The most useful changes were root-cause fixes with narrow validation:

- Seed propagation fixed a real reproducibility gap instead of tuning around noisy route-spread
  behavior.
- Static recenter fixed a concrete local-minimum pattern: forward rollout was rejected in a doorway,
  but rotating opened a safe slow-forward heading.
- Map-aware observation caps fixed a contract bug where the planner observed clipped goals on maps
  larger than the legacy `50 m` cap.
- Scenario-specific overrides avoided broad-family regressions. A broad Francis `0.6 m/s` override
  recovered perpendicular traffic but introduced failures in `francis2023_narrow_hallway` and
  `francis2023_join_group`; the scoped override kept those scenarios successful.
- ORCA clearly solves the remaining Francis leave-group dynamic hard-radius deadlock in targeted
  diagnostics.

## What Did Not Work

Constant-only tuning was generally weak or unsafe:

- `hybrid_rule_v3_fast_social_push`: `5/18` nominal successes, regressed heavily.
- `hybrid_rule_v3_waypoint2_static_escape`: `13/18` nominal successes with `3` static-collision
  terminations.
- Accepted-stop static reorientation: no aggregate improvement.
- `static_hard_safety_margin: 0.0` globally: `12/18` nominal successes and `4` static collisions.
- `hybrid_rule_v3_fast_progress_static015`: no aggregate improvement.
- `hybrid_rule_v4_recovery_aware`: worse than fast progress.
- Reverse static-clearance escape: no aggregate improvement and more doorway near-miss samples.
- Longer route-guide lookahead: no recovery on targeted crossing/doorway failures.
- Bounded static-clearance entry: either no recovery or converted timeout into obstacle collision.
- Deadlock rotation scoring variants: caused obstacle collisions or still timed out.
- Shorter rollout horizons on `francis2023_narrow_doorway`: `0.4 s` and `0.6 s` timed out;
  `0.8 s`, `1.0 s`, and `1.2 s` collided.
- Raising `static_recenter_weight` on `classic_merging_medium` seed `111`: converted timeouts into
  obstacle collisions around steps `304-315`.
- Lower speed/turn caps for merging: either collided or timed out, sometimes with many near misses.
- Scenario-scoped `static_hard_safety_margin: 0.0` for classic merging avoided one obstacle
  collision but still failed the tested merging seeds, ending with all candidates rejected by static
  clearance.

Existing non-local alternatives also did not solve the classic merging failures:

- Tuned ORCA solved `francis2023_leave_group`, but not `classic_merging_low` seeds `111`/`113` or
  `classic_merging_medium` seed `111`.
- `grid_route` and `planner_selector_v1` were useful probes in doorway cases but caused pedestrian
  collisions there, so they were not acceptable replacements for the retained hybrid candidate.

## Remaining Failure Analysis

### Classic Merging

The merging failures share a static-clearance local-minimum pattern. The route leads the robot into
a narrow clearance band where many moving candidates are rejected by `static_clearance`, and the
scorer often ranks zero-motion or tiny creep above useful escape. Relaxing static margin did not
solve the problem safely.

These failures remain policy work, not proven impossible:

- `classic_merging_low` seed `111`: obstacle collision at step `242` in the retained candidate.
- `classic_merging_low` seed `113`: timeout.
- `classic_merging_medium` seeds `111`, `112`, `113`: timeouts, with seed `113` also producing
  intrusive near-miss evidence in the full matrix.

### Francis Leave Group

The retained hybrid candidate freezes because all candidates are rejected by dynamic collision when
a pedestrian remains just inside the hard dynamic radius. Tuned ORCA resolves all three seeds in
targeted diagnostics, so a scenario-adaptive algorithm switch is promising. It still needs a full
matrix rerun before it can replace the current best fully validated candidate.

## Validation Commands

Focused final checks run before this wrap-up:

```bash
uv run ruff check robot_sf/benchmark/map_runner.py robot_sf/planner/hybrid_rule_local_planner.py \
  robot_sf/sensor/socnav_observation.py scripts/validation/run_policy_search_candidate.py \
  scripts/validation/run_policy_search_step_diagnostics.py tests/benchmark/test_map_runner_utils.py \
  tests/planner/test_hybrid_rule_local_planner.py tests/test_socnav_observation.py \
  tests/validation/test_run_policy_search_candidate.py

uv run ruff format --check robot_sf/benchmark/map_runner.py \
  robot_sf/planner/hybrid_rule_local_planner.py robot_sf/sensor/socnav_observation.py \
  scripts/validation/run_policy_search_candidate.py \
  scripts/validation/run_policy_search_step_diagnostics.py tests/benchmark/test_map_runner_utils.py \
  tests/planner/test_hybrid_rule_local_planner.py tests/test_socnav_observation.py \
  tests/validation/test_run_policy_search_candidate.py

uv run pytest tests/benchmark/test_map_runner_utils.py \
  tests/validation/test_run_policy_search_candidate.py \
  tests/planner/test_hybrid_rule_local_planner.py tests/test_socnav_observation.py -q
```

Results:

- Ruff check passed.
- Ruff format check passed.
- Candidate YAML validation passed.
- Focused pytest set passed: `106 passed`.

Additional runner override test after adding `scenario_algo_overrides`:

```bash
uv run pytest \
  tests/validation/test_run_policy_search_candidate.py::test_effective_candidate_runtime_applies_scenario_algo_override -q
```

Result: `1 passed`.

## Next Steps

1. Rerun `scenario_adaptive_hybrid_orca_v1` on `full_matrix` to confirm whether targeted
   leave-group success improves the aggregate from `130/141` to `131/141` without regressions.
2. Continue root-cause work on the classic merging family. Do not classify those failures as
   impossible without stronger evidence.
3. Avoid more broad constant sweeps unless a new planner mechanism or scenario-specific hypothesis
   explains why the change should recover the merging local minimum without weakening static
   safety.
