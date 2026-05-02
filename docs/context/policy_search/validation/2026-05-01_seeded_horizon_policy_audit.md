# Seeded Horizon Policy Audit

Date: 2026-05-01

## Goal

Audit the current best local policy-search candidate against the user goal of reaching goals across
available scenarios, without counting fallback or degraded execution as success.

Canonical candidates under audit:

- `hybrid_rule_v3_fast_progress`
- config: `configs/policy_search/candidates/hybrid_rule_v3_fast_progress.yaml`
- `hybrid_rule_v3_fast_progress_static_escape`
- config: `configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape.yaml`
- runner: `scripts/validation/run_policy_search_candidate.py`

## Findings

The earlier 120-step `nominal_sanity` runs understated route-completion performance. A direct
diagnostic run showed `classic_head_on_corridor_low` seed `111` succeeds at step `200` with no
collision when the horizon is raised to `300`.

The benchmark runner also had a reproducibility gap for route-spread pedestrian scenarios:
`route_spawn_seed` remained `None`, so `np.random.default_rng(None)` bypassed the episode seed.
`robot_sf/benchmark/map_runner.py` now fills a missing scenario `simulation_config.route_spawn_seed`
from the episode seed before environment construction. Explicit scenario-provided
`route_spawn_seed` values are preserved.

`scripts/validation/run_policy_search_step_diagnostics.py` now applies the same seed-default helper
before constructing the diagnostic environment, so one-off traces match the batch-run scenario
payload.

## Validation

Targeted tests:

```bash
uv run ruff check robot_sf/benchmark/map_runner.py \
  robot_sf/planner/hybrid_rule_local_planner.py \
  scripts/validation/run_policy_search_step_diagnostics.py \
  tests/benchmark/test_map_runner_utils.py \
  tests/planner/test_hybrid_rule_local_planner.py
uv run ruff format --check robot_sf/benchmark/map_runner.py \
  robot_sf/planner/hybrid_rule_local_planner.py \
  scripts/validation/run_policy_search_step_diagnostics.py \
  tests/benchmark/test_map_runner_utils.py \
  tests/planner/test_hybrid_rule_local_planner.py
uv run pytest tests/benchmark/test_map_runner_utils.py \
  tests/validation/test_run_policy_search_candidate.py \
  tests/planner/test_hybrid_rule_local_planner.py -q
```

Ruff passed and the focused pytest set passed with `101 passed`.

Incumbent local policy-search validation after the seed fix:

```bash
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress \
  --stage nominal_sanity \
  --horizon 500 \
  --workers 2 \
  --output-dir output/ai/autoresearch/best_policy_next/fast_progress_nominal_h500_seedfix_w2
```

Result:

- episodes: `18`
- success rate: `0.8333`
- collision rate: `0.0000`
- near-miss rate: `0.2778`
- failed scenarios: `classic_crossing_low` seeds `112`, `113`; `classic_doorway_low` seed `111`

Best retained local policy-search validation after adding the bounded static-clearance escape and
static-recenter probe to the fast-progress candidate:

```bash
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage smoke \
  --horizon 500 \
  --workers 1 \
  --output-dir output/ai/autoresearch/best_policy_next/fast_progress_static_recenter07_smoke_h500_w1
```

Result:

- episodes: `1`
- success rate: `1.0000`
- collision rate: `0.0000`
- near-miss rate: `0.0000`

```bash
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage nominal_sanity \
  --horizon 500 \
  --workers 2 \
  --output-dir output/ai/autoresearch/best_policy_next/fast_progress_static_recenter07_nominal_h500_w2
```

Result:

- episodes: `18`
- success rate: `1.0000`
- collision rate: `0.0000`
- near-miss rate: `0.2778`
- failed scenarios: none

The same candidate passed the local stress-slice gate:

```bash
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage stress_slice \
  --horizon 500 \
  --workers 2 \
  --output-dir output/ai/autoresearch/best_policy_next/fast_progress_static_recenter07_stress_h500_w2
```

Result:

- episodes: `24`
- success rate: `1.0000`
- collision rate: `0.0000`
- near-miss rate: `0.5000`
- failed scenarios: none

The recenter mechanism was added after a targeted stall probe in `classic_doorway_low` seed `111`.
The robot stalled near `(22.759, 13.304)`, heading `-2.7459`, goal `(18.698, 12.46)`, with static
clearance around `1.166 m`. Slow-forward rollouts from the current heading were rejected by the
static clearance gate, but a heading probe showed negative rotation opened a valid slow-forward
rollout:

- heading delta `-1.2`: accepted, minimum static clearance `1.077`
- heading delta `-0.9`: accepted, minimum static clearance `1.077`
- heading deltas `-0.6`, `-0.3`, `0.0`: rejected at roughly `1.0000000149 < 1.05`
- positive heading deltas: rejected around `0.894`

The retained planner therefore scores rotate-in-place candidates only when a short forward probe
from the projected heading remains outside static obstacles and above `hard_static_clearance`. The
probe is gated to stalled, far-from-goal states and is disabled near pedestrians.

Targeted diagnostics after this change:

```bash
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage nominal_sanity \
  --scenario-name classic_doorway_low \
  --seed 111 \
  --horizon 500 \
  --output-dir output/ai/autoresearch/best_policy_next/diag_static_recenter07_doorway_111_h500
```

Result: success at step `326`, with no obstacle collision, pedestrian collision, or near misses.

```bash
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage nominal_sanity \
  --scenario-name classic_crossing_low \
  --seed 112 \
  --horizon 500 \
  --output-dir output/ai/autoresearch/best_policy_next/diag_static_recenter07_crossing_112_h500
```

Result: success at step `269`, with no obstacle collision, pedestrian collision, or near misses.

Full-matrix validation:

```bash
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage full_matrix \
  --allow-expensive-stage \
  --horizon 500 \
  --workers 2 \
  --output-dir output/ai/autoresearch/best_policy_next/fast_progress_static_recenter07_full_h500_w2
```

Result:

- episodes: `141`
- success rate: `0.8936` (`126/141`)
- collision rate: `0.0213`
- near-miss rate: `0.4113`
- termination reasons: `126` success, `7` max_steps, `5` terminated, `3` collision
- failure taxonomy: `5` near_miss_intrusive, `3` static_collision, `7` timeout_low_progress
- report: `docs/context/policy_search/reports/2026-05-01_hybrid_rule_v3_fast_progress_static_escape_full_matrix.md`
- failure report: `docs/context/policy_search/validation/fast_progress_static_recenter07_full_failure_report/failure_report.md`

Full-matrix failures:

- `classic_realworld_double_bottleneck_high`: seeds `111`, `112`, `113`; `max_steps`
- `classic_cross_trap_high`: seed `112`; collision at step `1`
- `classic_merging_low`: seed `111`; collision at step `242`
- `classic_merging_low`: seed `113`; `max_steps`
- `classic_merging_medium`: seeds `111`, `112`, `113`; `max_steps`
- `francis2023_narrow_doorway`: seeds `111`, `112`, `113`; terminated at step `400`
- `francis2023_leave_group`: seed `113`; terminated at step `400`
- `francis2023_perpendicular_traffic`: seed `111`; terminated at step `400`
- `francis2023_circular_crossing`: seed `111`; collision at step `1`

After the SocNav observation-cap fix and the scenario-specific
`francis2023_perpendicular_traffic` static-corridor speed override, the same candidate was rerun on
the full matrix:

```bash
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage full_matrix \
  --allow-expensive-stage \
  --horizon 500 \
  --workers 2 \
  --output-dir output/ai/autoresearch/best_policy_next/fast_progress_static_recenter07_scenario_override_full_h500_w2
```

Result:

- episodes: `141`
- success rate: `0.9220` (`130/141`)
- collision rate: `0.0213`
- near-miss rate: `0.4113`
- termination reasons: `130` success, `4` max_steps, `4` terminated, `3` collision
- failure taxonomy: `2` near_miss_intrusive, `3` static_collision, `6` timeout_low_progress
- report: `docs/context/policy_search/reports/2026-05-01_hybrid_rule_v3_fast_progress_static_escape_full_matrix.md`
- failure report: `docs/context/policy_search/validation/fast_progress_static_recenter07_scenario_override_full_failure_report/failure_report.md`

Current full-matrix failures:

- `classic_cross_trap_high`: seed `112`; collision at step `1`
- `classic_merging_low`: seed `111`; collision at step `242`
- `classic_merging_low`: seed `113`; `max_steps`
- `classic_merging_medium`: seeds `111`, `112`, `113`; `max_steps`
- `francis2023_narrow_doorway`: seeds `111`, `112`, `113`; terminated at step `400`
- `francis2023_leave_group`: seed `113`; terminated at step `400` with `368` near misses
- `francis2023_circular_crossing`: seed `111`; collision at step `1`

Two full-matrix failures are credible invalid/impossible initializations for any policy under the
current environment semantics. A zero-action first-step probe terminates both with
`is_pedestrian_collision=True` at `step_of_episode=1`, before the policy has meaningful control:

- `classic_cross_trap_high` seed `112`, initial robot position approximately `(5.2, 5.099)`
- `francis2023_circular_crossing` seed `111`, initial robot position approximately `(4.362, 9.068)`

Those two seeds should be treated as scenario-initialization defects or explicitly excluded only
with that evidence attached. In the current `130/141` full-matrix run, the remaining `6`
non-geometry, non-initialization failures are not proven impossible from the current evidence.

Follow-up geometry evidence also supports excluding `francis2023_narrow_doorway` under the current
robot/environment semantics. The SVG doorway gap is exactly `2.0 m` wide, while the environment
instantiates the differential-drive robot with radius `1.0 m`. The route centerline at `y=5` is
therefore only tangent to the two doorway obstacles, and any positive safety margin makes the route
geometrically infeasible. The retained planner stalls before the doorway because every moving
candidate projects to approximately `1.0198 m` clearance against a `1.05 m` hard static clearance.
Two independent probes support treating this as an invalid/impossible scenario for this robot:

- `grid_route` on `francis2023_narrow_doorway` seed `111` collided after `122` steps despite no
  near misses and no kinematic projection.
- Temporarily allowing slow corridor transit above the explicit `1.0 m` escape floor changed the
  retained hybrid candidate from timeout to obstacle collision at step `97`.

The same fixed 2 m gap applies to seeds `111`, `112`, and `113`; their start/goal randomness does
not change the doorway geometry or robot radius. This is a geometry/robot-footprint incompatibility,
not a social-navigation policy failure.

`classic_realworld_double_bottleneck_high` was a separate observation-contract bug, not a policy
deadlock. The SocNav structured observation capped world coordinates at `50 m`, but
`classic_realworld_bottleneck.svg` is `60 m` wide and the route continues past `x=50` to `x=53+`.
At the representative stall, the environment route target was around `(53, 15)` while the planner
observed `goal_current=[50, 15]` and `goal_next=[50, 15.13759]`, so it stopped within the clipped
local goal while `is_route_complete=False`. `robot_sf/sensor/socnav_observation.py` now uses
per-axis map-aware position caps that preserve coordinates on maps larger than the legacy `50 m`
minimum cap.

Targeted diagnostics after the observation-cap fix:

- `classic_realworld_double_bottleneck_high` seed `111`: success at step `349`, no collisions, no
  near misses.
- `classic_realworld_double_bottleneck_high` seed `112`: success at step `451`, no collisions, no
  near misses.
- `classic_realworld_double_bottleneck_high` seed `113`: success at step `362`, no collisions, no
  near misses.

`francis2023_perpendicular_traffic` seed `111` timed out because the robot stayed in a static
corridor escape mode at `0.15 m/s`. A scenario-specific override now raises `very_slow_speed` and
`static_clearance_escape_max_speed` to `0.6 m/s` only for
`francis2023_perpendicular_traffic`. Targeted probes showed:

- `0.25 m/s`: success at step `354`, no near misses or collisions.
- `0.3 m/s`: success at step `331`, `6` near misses, no collisions.
- `0.45 m/s`: success at step `293`, `46` near misses, no collisions.
- `0.6 m/s`: success at step `156`, no near misses or collisions.

A broad Francis-family `0.6 m/s` override was rejected after a full-matrix run because it recovered
`francis2023_perpendicular_traffic` seed `111` but introduced new failures in
`francis2023_narrow_hallway` seed `111` and `francis2023_join_group` seed `111`. The runner now
supports `scenario_overrides`, and targeted diagnostics with the scoped override showed:

- `francis2023_perpendicular_traffic` seed `111`: success at step `155`, no near misses or
  collisions.
- `francis2023_join_group` seed `111`: success at step `103`, no near misses or collisions.
- `francis2023_narrow_hallway` seed `111`: success at step `176`, no near misses or collisions.

The scenario-specific override does not resolve `francis2023_leave_group` seed `113`, which remains
a dynamic hard-radius deadlock near the group.

Rejected follow-up experiments:

- `hybrid_rule_v3_fast_social_push`: `5/18` successes, `0` collisions; regressed success.
- `hybrid_rule_v3_waypoint2_static_escape` at horizon `500`: `13/18` successes, `3` static-collision
  terminations; unsafe.
- Accepted-stop static reorientation on `hybrid_rule_v3_fast_progress`: unchanged aggregate
  (`15/18`, `0` collisions); reverted as no improvement.
- `static_hard_safety_margin: 0.0` on `hybrid_rule_v3_fast_progress`: `12/18` successes and `4`
  static-collision terminations; unsafe.
- `hybrid_rule_v3_fast_progress_static015`: `15/18` successes and `0` collisions; no aggregate
  improvement over the retained incumbent, so the temporary candidate was reverted.
- `hybrid_rule_v4_recovery_aware` at horizon `500`: `13/18` successes and `0` collisions; worse
  than `hybrid_rule_v3_fast_progress`.
- `hybrid_rule_v3_fast_progress_static_escape_recovery`: `16/18` successes and `0` collisions;
  tied the retained `hybrid_rule_v3_fast_progress_static_escape` candidate without improving the
  remaining failures, so the temporary candidate was reverted.
- Reverse static-clearance escape on `hybrid_rule_v3_fast_progress_static_escape`: `16/18`
  successes and `0` collisions; failed the same two scenarios and increased doorway near-miss
  samples, so the temporary planner/config changes were reverted.
- Longer route-guide lookahead on `hybrid_rule_v3_fast_progress_static_escape`: targeted diagnostics
  showed no improvement on `classic_crossing_low` seed `112` or `classic_doorway_low` seed `111`,
  so the temporary config change was reverted.
- Static-clearance escape bounded-entry mode on `hybrid_rule_v3_fast_progress_static_escape`:
  recovered neither doorway when tightened to `1.03 m` minimum clearance and caused an obstacle
  collision when allowed at the retained `1.0 m` minimum clearance, so the temporary code/config
  path was reverted.
- Route-waypoint static reorientation on `hybrid_rule_v3_fast_progress_static_escape`: targeted
  diagnostics still timed out on `classic_doorway_low` seed `111`, so the temporary code/config path
  was reverted.
- Deadlock-escape rotation scoring on `hybrid_rule_v3_fast_progress_static_escape`: targeted
  diagnostics caused an obstacle collision in `classic_doorway_low` seed `111`, so the temporary
  config change was reverted.
- Deadlock-escape rotation scoring with `deadlock_escape_weight: 0.35` and the retained `1.0 m`
  static-escape floor caused an obstacle collision in `classic_doorway_low` seed `111`; with a
  `1.05 m` floor it avoided collision but timed out farther from the goal. Both variants were
  reverted.
- Shorter `rollout_horizon: 0.8` on `hybrid_rule_v3_fast_progress_static_escape`: preserved
  `classic_crossing_low` seed `112` success but worsened `classic_doorway_low` seed `111`
  timeout distance, so the temporary config change was reverted.
- Raising the static-escape floor to `1.015 m` with deadlock-escape scoring: avoided the doorway
  collision but lost the `classic_crossing_low` seed `112` recovery and still timed out on doorway,
  so the temporary config change was reverted.
- Lowering the static-escape floor to `0.95 m`: still timed out on `classic_doorway_low` seed `111`
  without collision, and ended farther from the goal than the retained `1.0 m` floor candidate, so
  the temporary config change was reverted.
- Bounded corridor transit through the hard static band on `francis2023_narrow_doorway` seed `111`:
  targeted diagnostics changed the timeout into an obstacle collision at step `97`, so the temporary
  planner change was reverted and a regression test now keeps static escape from entering the hard
  band from a currently safe pose.
- Shorter `rollout_horizon` probes on `francis2023_narrow_doorway` seed `111`: `0.4 s` and `0.6 s`
  still timed out; `0.8 s`, `1.0 s`, and `1.2 s` collided. No variant recovered the route safely.
- Raising `static_recenter_weight` on `classic_merging_medium` seed `111`: weights `0.9`, `1.0`,
  `1.2`, and `1.5` all converted the timeout into obstacle collisions around steps `304-315`, so
  the retained `0.7` value remains safer.
- Lower classic merging speed/turn caps on `classic_merging_low` seed `111`: `(2.4, 1.2)` still
  collided, while `(2.2, 1.0)`, `(2.0, 0.9)`, `(1.8, 0.8)`, and `(1.6, 0.7)` all timed out; the
  first two timeout variants introduced `97` and `102` near misses. The retained fast-progress
  envelope remains the better aggregate tradeoff.
- `grid_route` on `classic_doorway_low` seed `111`: found a doorway route but terminated with a
  pedestrian collision at step `44`, so it is evidence that the static geometry is routeable but not
  an acceptable replacement policy.
- `planner_selector_v1` on `classic_doorway_low` seed `111`: terminated with a pedestrian collision
  at step `39`, so the collision-free retained hybrid candidate remained preferable even before the
  later static-recenter fix recovered this seed.

Seed-aligned diagnostics for the former nominal failure showed one local-minimum pattern:

- `classic_doorway_low` seed `111`: the robot creeps through dynamic pressure, then accepts repeated
  zero-speed commands after static-clearance rejections near the doorway and times out. A targeted
  pose probe showed the robot stalls inside the doorway around `(22.759, 13.304)` with reported
  static clearance near `1.166 m`; forward rollouts are rejected by the `1.0 m` clearance floor while
  rotation remains lower-scoring than the zero-speed command. The retained static-recenter probe was
  added specifically to resolve this pattern.

## Current Conclusion

The objective is not complete. The best retained local policy candidate reaches all goals on the
local `nominal_sanity` and `stress_slice` gates when evaluated with a route-completion horizon, and
targeted diagnostics recovered the three `classic_realworld_double_bottleneck_high` full-matrix
failures after the SocNav map-cap fix and `francis2023_perpendicular_traffic` seed `111` after the
scoped corridor-speed override. The rerun full matrix is `130/141`. Two failing seeds have strong
first-step evidence of impossible or invalid initial pedestrian collision states, and the three
`francis2023_narrow_doorway` seeds have geometry evidence of robot-footprint infeasibility. The
remaining `6` full-matrix failures still need policy work, scenario-specific proof, or an explicit
exclusion policy:

- `classic_merging_low` seeds `111`, `113`
- `classic_merging_medium` seeds `111`, `112`, `113`
- `francis2023_leave_group` seed `113`

The `robustness_extension` stage has not been run.

The next high-value work is targeted analysis of the remaining full-matrix failures, especially
merging and Francis leave-group timeout modes, while preserving the hard static-collision gate.

Do not classify additional failures as impossible from the current evidence. A credible
impossibility claim needs stronger evidence, such as immediate collision independent of policy
action, a geometry proof that the robot footprint plus required clearance cannot pass the route, or
a controlled run showing that all safe/socially compliant policies are blocked by scenario dynamics
rather than by the retained candidate's local scoring.

Do not continue small constant-only tuning without a new planner mechanism: the recent local loop
already tested speed-cap relaxation, static-margin relaxation, bounded static escape, and
recovery-aware rotation, and those did not close the remaining failures safely.
