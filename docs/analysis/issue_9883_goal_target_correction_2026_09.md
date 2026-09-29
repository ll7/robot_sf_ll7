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
target is reached (`robot_sf/nav/navigation.py:569-588`). Unknown
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

The owner of #9751's 0.0.8 campaign template must replace only the Risk-DWA
and Predictive-MPPI `algo_config` paths with the corresponding
`*_camera_ready_goal_v2.yaml` paths after this PR is reviewed. A proposed
0.0.8 row using either historical path is uncorrected and must fail admission.
The currently separate #9879 release-candidate path hashes algo configs but
does not enforce this semantic requirement. A read-only assertion against the
current campaign template exits 1 and reports both violations:

```text
risk_dwa: expected configs/algos/risk_dwa_camera_ready_goal_v2.yaml;
          found configs/algos/risk_dwa_camera_ready.yaml
predictive_mppi: expected configs/algos/predictive_mppi_camera_ready_goal_v2.yaml;
                 found configs/algos/predictive_mppi_camera_ready.yaml
```

Before #9883 can close, the #9879 candidate admission path needs a focused
assertion that each of those arm keys occurs once, uses the exact corrected
path, and parses to `goal_target_version: active_waypoint_v2`. Its regression
test must reject the present old-path template and admit an otherwise
identical corrected copy. This dependency is pending; the new configs alone
do not enforce it.
The final campaign manifest must pin the corrected source/config hashes and
the 14-arm, 48-identity, seed-111–140 contract. Release preflight, complete
model-backed evaluation, and the paired 0.0.7/0.0.8 episode comparison remain
required. Attribute any row differences to the versioned selector only where
traces and inputs support that cause; no rate or ranking impact is asserted
here. Keep the 0.0.7 configs, release bundle, tag, and record byte-identical.
