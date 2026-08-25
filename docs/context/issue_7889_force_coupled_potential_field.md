# Force-Coupled Potential-Field Planner Method Card (`force_coupled_potential_field`)

**Status:** experimental / opt-in / smoke-only — implementation-integrity and deterministic-smoke
proof; not a faithful reproduction, not benchmark evidence, not a release-roster change.
**Issue:** [#7889](https://github.com/ll7/robot_sf_ll7/issues/7889) (parent research
[#7882](https://github.com/ll7/robot_sf_ll7/issues/7882), programme parent
[#7319](https://github.com/ll7/robot_sf_ll7/issues/7319)).
**Owner module:** `robot_sf/planner/force_coupled_potential_field.py`.
**Config:** `configs/algos/issue_7889_force_coupled_potential_field.yaml`.
**Source method:** Jing et al., "Local path planning for autonomous vehicles: a dynamic potential
field-guided and force-coupled adaptive pure pursuit approach" (Scientific Reports 2026).

Plain-language summary: an opt-in local planner that steers a unicycle robot by combining an
attractive force toward a look-ahead target with repulsive forces from obstacles and pedestrians,
saturating the total force, and issuing a speed/steering command that respects configured speed and
rate limits as hard predicates. It is a comparator core for future planner evaluation — nothing
more.

## Source-to-implementation map

| Source element | Implementation | Notes |
| --- | --- | --- |
| velocity-adaptive local path / look-ahead selection | `implemented` (approximated) | look-ahead distance scales with remaining goal distance, clamped to `[look_ahead_min_m, look_ahead_max_m]` |
| dynamic potential-field refinement | `approximated` | classic attractive + inverse-distance repulsive field; no time-varying field update |
| force-coupled target selection | `implemented` | combined saturated force direction sets the desired heading |
| bounded forward-kinematic command generation | `implemented` | unicycle `(linear, angular)` with speed limits |
| steering/angular-rate and rate-of-change constraints | `implemented` | speed and command-rate limits enforced as clips (hard predicates) |

## Formulae and conventions

- Attractive force: `attractive_weight * (target - robot) / |target - robot|`.
- Repulsive force per source point: `repulsive_weight * (1/d - 1/influence_radius_m) * (robot - p) / d`
  for `d <= influence_radius_m`; zero-distance points push along `+x` (guarded, finite).
- Total force saturated at `force_saturation` magnitude.
- Desired heading: `atan2(fy, fx)`; command `linear = look_ahead_gain * goal_distance`,
  `angular = wrap_pi(desired_heading - robot_theta)`.
- Units: metres, seconds, radians. Sign convention: left-hand positive angular rate.
- Numerical guards: `numerical_epsilon` for zero-distance and division; non-finite inputs raise.

## Deviations and unsupported elements

- No time-varying potential field (source's dynamic refinement is approximated by static fields).
- No steering-rate *acceleration* profile beyond the configured per-step rate limits.
- No pedestrian motion prediction; pedestrians repel at their observed positions only.
- No global route integration; the local target is derived from the goal directly.
- The source paper's vehicle-oriented results are not transferred as social-navigation evidence.

## Deterministic smoke

- Command and diagnostics are deterministic for a fixed observation sequence and seed
  (`reset(*, seed=...)`), excluding timestamps.
- Test fixtures: goal approach, obstacle repulsion, pedestrian repulsion, symmetric obstacle
  tie-break, zero-distance guard, speed/rate hard-predicate bounds, rotation/translation
  consistency.
- Outcome status: `ok` on success; `ValueError` (fail closed) on missing/non-finite inputs.
