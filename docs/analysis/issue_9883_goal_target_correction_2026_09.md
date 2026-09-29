# #9883: versioned active-waypoint selection for 0.0.8

Status: **diagnostic correction candidate; no benchmark row admitted**. This
report records the causal source path and a small deterministic route replay.
It does not estimate success, collision, or ranking effects.

## Contract and version boundary

The simulator exposes each route navigator's current and next waypoint
separately (`robot_sf/sim/simulator.py:1649-1657`). The SocNav producer writes
`[0, 0]` when no next waypoint exists
(`robot_sf/sensor/socnav_observation.py:931-936`); the observation contract
calls `goal.current` current and `goal.next` next
(`docs/dev/observation_contract.md:108-109`). Before this change, Risk-DWA and
Predictive-MPPI chose `goal.next` whenever it differed from the robot's
position. On a route ending away from the origin, that made both optimize
toward the absent-next sentinel. With a real next waypoint, they skipped the
unreached current one. The nested Predictive-MPPI predictor already uses
`goal.current` (`robot_sf/planner/socnav_prediction.py:1722-1725`), so the
optimizer and its anchor could disagree.

`robot_sf/planner/goal_target.py:7-37` now gives both planners an explicit
`goal_target_version`. The default `legacy_next_goal_v1` reproduces the old
selection formula. `active_waypoint_v2` always targets `goal.current`;
`RouteNavigator.update_position` advances its waypoint index when the current
target is reached (`robot_sf/nav/navigation.py:603-622`). Unknown
selectors raise at config construction. The two existing camera-ready YAMLs
remain untouched. The new `*_camera_ready_goal_v2.yaml` files differ in parsed
values only by `goal_target_version: active_waypoint_v2`.

| Config | SHA-256 |
| --- | --- |
| `configs/algos/risk_dwa_camera_ready.yaml` | `1351439539ed02334891f6a1ef6ac652745791b1b830a5f8616850fbd5ccb37c` |
| `configs/algos/risk_dwa_camera_ready_goal_v2.yaml` | `25c00dbaebfd1e8be07e93cfd967a673c2abed79412e6979b470f788035e7daa` |
| `configs/algos/predictive_mppi_camera_ready.yaml` | `5213a9e44c74cb79f78466d414645f6ca233d761e8fa5b77e5cfc3161ed94adb` |
| `configs/algos/predictive_mppi_camera_ready_goal_v2.yaml` | `5d56d241d19f2f756d6cbc64cfdd46b63fe5ff5579bf0e3ce7413fe38cc83342` |

## Diagnostic route replay

An empty, obstacle-free structured observation starts the robot at `(5, 5)`
with heading 0. The intermediate stage has current `(8, 5)` and next `(8, 8)`.
The final stage has current `(8, 5)` and next sentinel `(0, 0)`. A third sample
places the robot exactly at the intermediate waypoint while the navigator has
not yet advanced. Risk-DWA uses its camera-ready defaults. Predictive-MPPI
uses a fixed seed, eight samples, one CEM iteration, and a stub that returns
empty pedestrian futures and a fixed anchor; **its predictor is not loaded**.
The focused tests reproduce these target and command paths. Commands below
are `(linear m/s, angular rad/s)` from one planner call, not episode outcomes.

| Planner | Sample | Legacy target / command | V2 target / command |
| --- | --- | --- | --- |
| Risk-DWA | intermediate | `(8,8)` / `(1.2,0.8)` | `(8,5)` / `(1.2,0.0)` |
| Risk-DWA | at intermediate | `(8,8)` / `(1.2,1.2)` | `(8,5)` / `(0.0,0.0)` |
| Risk-DWA | final sentinel | `(0,0)` / `(0.0,-1.2)` | `(8,5)` / `(1.2,0.0)` |
| Predictive-MPPI | intermediate | `(8,8)` / `(0.55,1.1780972451)` | `(8,5)` / `(0.55,0.0)` |
| Predictive-MPPI | at intermediate | `(8,8)` / `(0.55,1.3)` | `(8,5)` / `(0.0,0.0)` |
| Predictive-MPPI | final sentinel | `(0,0)` / `(0.55,-1.3)` | `(8,5)` / `(0.55,0.0)` |

The structured and flat zero-sentinel cases, intermediate route advancement,
legacy default, unknown-version failure, and parsed YAML equivalence are
asserted in `tests/planner/test_risk_dwa.py` and
`tests/planner/test_predictive_mppi_planner.py`. The sampled command changes
show that historical and corrected results need a causal comparison. The
Predictive-MPPI replay is diagnostic because its predictor is stubbed; it
cannot count as model-provenance or benchmark evidence.

## Release integration and open gates

The 0.0.8 campaign template now binds Risk-DWA and Predictive-MPPI to their
`*_camera_ready_goal_v2.yaml` paths, and guarded PPO to its v2 fallback
profile. Candidate admission checks unique planner keys, those exact paths,
and the effective `active_waypoint_v2` selector in all three arms. Each arm
has a historical-path and selector-mutation regression test. This is static
candidate proof, not a release or episode outcome.

Guarded PPO has a deliberate one-step boundary difference: when the robot is
already within `goal_tolerance` of `goal.current`, its outer guard looks ahead
to `goal.next`, while its configured Risk-DWA fallback still targets
`goal.current`. The simulator advances the active waypoint on the next step.
`test_guarded_ppo_outer_guard_looks_ahead_for_one_waypoint_boundary_step`
pins both targets without changing the guard's method. The active-waypoint
claim for guarded PPO applies to its fallback, not its outer guard.
The final campaign manifest must pin the corrected source/config hashes and
the 14-arm, 48-identity, seed-111–140 contract. Release preflight, complete
model-backed evaluation, and the paired 0.0.7/0.0.8 episode comparison remain
required. Attribute any row differences to the versioned selector only where
traces and inputs support that cause; no rate or ranking impact is asserted
here. Keep the 0.0.7 configs, release bundle, tag, and record byte-identical.
