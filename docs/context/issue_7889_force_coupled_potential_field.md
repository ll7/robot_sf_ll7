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
rate limits as hard predicates. The registered map-runner path reads static obstacles from the
canonical occupancy-grid obstacle channel; direct point inputs remain available for analytic smoke
fixtures. It is a comparator core for future planner evaluation — nothing more.

## Source-to-implementation map

| Source element | Implementation | Notes |
| --- | --- | --- |
| goal-distance-adaptive local path / look-ahead selection | `implemented` (approximated) | look-ahead distance scales with remaining goal distance, not velocity, and is clamped to `[look_ahead_min_m, look_ahead_max_m]` |
| dynamic potential-field refinement | `approximated` | classic attractive + inverse-distance repulsive field; no time-varying field update |
| force-coupled target selection | `implemented` | combined saturated force direction sets the desired heading |
| bounded forward-kinematic command generation | `implemented` | unicycle `(linear, angular)` with speed limits |
| steering/angular-rate and rate-of-change constraints | `implemented` | speed and command-rate limits enforced as clips (hard predicates) |

## Formulae and conventions

- Attractive force: `attractive_weight * (target - robot) / |target - robot|`.
- Repulsive force per source point: `repulsive_weight * (1/d - 1/influence_radius_m) * (robot - p) / d`
  for `numerical_epsilon < d <= influence_radius_m`.
- A zero-distance obstacle or pedestrian is treated as an overlap: the planner emits an immediate
  `(0, 0)` stop and records sticky `status: degraded`; it does not invent a repulsion direction.
- Reaching the goal emits a nominal immediate `(0, 0)` stop. A near-zero total force away from the
  goal is treated as a potential-field local minimum: the planner stops, records sticky
  `status: degraded`, and does not pass the undefined `atan2(0, 0)` direction into control.
- Total force saturated at `force_saturation` magnitude.
- Desired heading: `atan2(fy, fx)`; command `linear = look_ahead_gain * goal_distance`,
  `angular = wrap_pi(desired_heading - robot_theta)`.
- Units: metres, seconds, radians. Sign convention: left-hand positive angular rate.
- Static-obstacle input: direct world-frame points for analytic fixtures, otherwise occupied cell
  centres from the map runner's canonical occupancy-grid obstacle channel, thresholded and bounded
  to the nearest configured point count inside the influence radius. The explicit obstacle channel
  and its finite, shape-consistent metadata are required; combined occupancy is not treated as a
  static-obstacle substitute.
- Numerical guards: `numerical_epsilon` for overlap, division, goal arrival, and total-force
  cancellation; non-finite inputs raise.

## Deviations and unsupported elements

- No time-varying potential field (source's dynamic refinement is approximated by static fields).
- No steering-rate *acceleration* profile beyond the configured per-step rate limits.
- No pedestrian motion prediction; pedestrians repel at their observed positions only.
- No global route integration; the local target is derived from the goal directly.
- The source paper's vehicle-oriented results are not transferred as social-navigation evidence.

## Deterministic smoke

The receipt below is implementation-integrity evidence only. It does not measure collision,
time-to-collision, comfort, social compliance, planner ranking, or fidelity to the source paper.
It is retained as a historical receipt for the named source commit and is not exact-head evidence
after the fail-closed occupancy/overlap corrections.

- Code/source commit: `3c3cca655652050c97771f10b7176cb0a07ebad8`.
- Comparison base: `31552ec8d3f2e963ba857b34b0efe65f49311fc9` (`origin/main`).
- Config digest (SHA-256):
  `ba14509401c6eb4b78c2f9af7575a0c8422b2eb374e3fc8d488c28d85c1340c3`.
- Environment: Python 3.12.8, NumPy 2.4.6,
  Linux 6.8.0-87-generic x86_64, glibc 2.39.
- Exact fixed-smoke command:

  ```bash
  scripts/dev/run_worktree_shared_venv.sh --profile all-extras -- uv run --no-sync pytest -q \
    tests/planner/test_force_coupled_potential_field.py::test_fixed_smoke_scenarios
  ```

| Scenario ID | Seed | Observed command `(m/s, rad/s)` | Status | Outcome |
| --- | ---: | --- | --- | --- |
| `analytic_static_obstacle` | 1 | `(0.16, -0.30)` | `ok` | deterministic success |
| `analytic_pedestrian_interaction` | 7 | `(0.16, -0.30)` | `ok` | deterministic success |

Both cases reported `linear_rate_limit` and `angular_rate_limit` as active on their first step,
with no fallback, degradation, or zero-distance guard. Repeated seeded replay produced identical
commands and diagnostics.

### Outcome-state boundary

| State | Receipt classification | Evidence or limitation |
| --- | --- | --- |
| success | observed | both fixed analytic fixtures returned finite, bounded commands with `status: ok` |
| invalid | observed | missing, malformed, and non-finite required inputs raise `ValueError` after recording `status: invalid_input` |
| degraded | observed | absent optional visibility or a zero-distance overlap is explicit and sticky for the episode as `status: degraded`; overlap emits an immediate stop and is not nominal success or benchmark evidence |
| collision | not evaluated | analytic command smoke has no simulator rollout, so no collision outcome is claimed |
| timeout | not evaluated | bounded analytic execution has no timeout outcome; no campaign/runtime claim is made |

The full focused validation also covers goal approach, separate obstacle and pedestrian force
components, symmetric tie-breaking, zero-distance stop semantics, sticky episode degradation, hard
speed/rate predicates, goal and force-cancellation stops, rotation/translation consistency,
canonical lifecycle behavior, and occupancy-grid-backed opt-in map-runner registration.
