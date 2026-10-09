# Opt-in footprint metrics for 0.1.0

Issues [#10079](https://github.com/ll7/robot_sf_ll7/issues/10079) and
[#10035](https://github.com/ll7/robot_sf_ll7/issues/10035) are implemented by the
separate `robot-sf-footprint.v1` definition. Enable it on a map scenario:

```yaml
metadata:
  footprint_metric_schema_version: robot-sf-footprint.v1
```

The runner writes `metrics.footprint_metrics.schema_version`. Existing top-level
`robot-sf-metrics.v2` scalar names, the legacy metric-definition digest, simulator
outcomes and SNQI inputs retain their released meanings. Frozen configurations
and release assets are not migrated. Aggregation rejects cohorts that mix the
opt-in block with legacy-only rows, or carry an unknown block version.

## Definitions

Each name below is inside `metrics.footprint_metrics`. Counts count samples, not
unique encounters. Safety geometry uses post-step positions. Radii come from the
resolved episode configuration; other-agent metrics require one explicit radius
per agent. A wall source must supply complete segments, including map bounds;
a sampled point cloud cannot substitute for it.

| Name | Definition and units | Missing-data rule |
| --- | --- | --- |
| `wall_collisions` | Number of samples with centre-to-segment distance minus robot radius <= 0 | Empty wall set: 0; missing segments with nonempty sampled walls: error |
| `agent_collisions` | Number of samples with any agent centre distance minus both radii <= 0 | No agents: 0; missing agent radii: error |
| `time_to_collision_min` | First constant-velocity disc contact, minimized across samples and pedestrians, seconds | No predicted contact: undefined |
| `space_compliance` | Fraction of all samples with any pedestrian surface gap < 0.50 m; this is a violation fraction | Entire episode without present pedestrians: undefined |
| `shortest_path_len` | Map-aware reference to the frozen completion target, with the centre domain shrunk by the robot radius, metres | Missing geometry, overlapping reset, or no positive-clearance route: undefined; `reference_status=unavailable` |
| `path_efficiency` | Reference / travelled path including reset, on successful collision-free episodes | Failure or unavailable reference: undefined; no clipping |
| `time_to_goal_ideal_ratio` | Actual completion time / (reference / maximum robot speed) | Failure, unavailable reference, or unavailable speed: undefined |
| `force_exceed_events` | Count of present pre-integration pedestrian-force samples above the existing comfort force threshold | Missing paired inputs: undefined; malformed present inputs: error |
| `comfort_exposure` | Above-threshold paired samples / present force samples | Empty population: 0; missing paired inputs: undefined |
| `force_exceed_near_robot` | Above-threshold paired samples with simultaneous pre-integration surface gap < 0.50 m | Missing paired inputs: undefined |

Undefined values are omitted by the existing JSON sanitizer. `reference_status`
and `force_pairing` distinguish unavailable input from a numeric zero.
`legacy_comparison` records the old reference, centre-based space violation and
post-step force/proximity join for the diagnostic before/after table.

### Disc contact TTC

For displacement `d = pedestrian - robot`, relative velocity
`v = robot_velocity - pedestrian_velocity`, and combined radius `R`, solve
`|d - v*t| = R`. Existing overlap/contact gives zero; receding pairs, stationary
separated pairs and trajectories that pass beside the disc have no contact TTC.
Pedestrian velocities use backward differences at samples 1..T-1; the first
sample has no derived pedestrian velocity and is excluded. Teleport/respawn
velocity identification remains the trajectory producer's responsibility.

The opt-in `near_miss_ttc` block counts contact predictions below 1, 2 and 3 s.
Calling the existing TTC diagnostic with opted-in `EpisodeData` also selects
contact geometry and declares `near_miss_ttc__definition=robot-sf-footprint.v1`.
Unmarked diagnostic calls preserve the old centre-distance/full-speed result.
No TTC threshold is promoted into the canonical distance-based near-miss metric.

### Reference geometry

The continuous visibility graph uses circumscribed 32-edge corner approximations
(radius `(robot_radius + 1e-8) / cos(pi/32)`), with exact segment-to-solid distance
checks. Exact checks accept clear starts inside the approximation band and
reject routes that need wall contact, including an aperture exactly one robot
diameter wide. Map boundaries shrink by the physical radius. Goal-zone entry
retains the simulator's centre-entry target; it does not erode the goal zone.
The waypoint completion reference remains the sampled point target. Radius,
original and expanded obstacle geometry, domain, reset and goal geometry enter
the cache identity. References are conservative polygon approximations, rather
than an exact analytic circular-arc geodesic.

The radius-free reference remains available through the existing default API.
Corrected references and time terms need their own reviewed normalization before
a score consumes them; this change does not recalibrate or replace SNQI.

### Force pairing and radius compatibility

`algorithm_metadata.robot_force_samples`, when step tracing is enabled, adds
`robot_pos`, `total_forces` and `force_pairing=pre-integration-inputs.v1` on opted-in
runs. Pedestrian positions come from the simulator's already captured force
inputs, after behaviour updates and before integration. The robot pose is copied
before `env.step`; its pose is unchanged during pedestrian behaviour updates.
Marginal force counts are usually unchanged by this timing repair. The spatial
join uses the input positions, never a one-frame shift across respawns. Variable
populations use explicit sample presence, excluding padding from the denominator.

Legacy force arrays retain their post-step layout. Robot-attributable component
forces already carry pre-integration inputs on main; they are not redefined.
Physical pedestrian radius and the historical force-kernel radius remain distinct
parameters. [PR #10075](https://github.com/ll7/robot_sf_ll7/pull/10075) owns the
shared pedestrian override. This implementation does not duplicate it. The
standalone runner's opted-in radius fallbacks come from physical robot/simulation
settings; its historical 0.30/0.35 m fallbacks remain for unmarked calls.

## Diagnostic readers and sensitivity

An opted-in simulation trace carries the definition marker plus `robot_radius_m`
and `ped_radius_m`. Critical intervals then use the surface gap for
`min_clearance_m`, positive gaps below 0.50 m for `near_miss_count`, gap <= 0 for
`collision_flag`, and disc contact TTC. The exported report declares both the
footprint definition and `ttc_convention=disc_contact_seconds.v1`. `min_distance_m`
stays centre distance. Unmarked interval inputs preserve historical fields.

Trace figures with opted-in metadata derive the centre-distance collision
reference from the sum of radii, and the comfort reference from that sum + 0.50 m.
Unmarked figures retain their historical default lines.
The generic trace-series exporter preserves the footprint marker and physical
radii for this reader. Unknown definitions or incomplete opted-in geometry
fail before any bundle is written.

The block reports near-miss gap counts at 0.10/0.20/0.30/0.50 m and comfort-gap
violation fractions at 0.30/0.50/0.80/1.20 m. These are descriptive sensitivity
surfaces, not empirically calibrated safety limits. The released 0.50 m SNQI gap
input is unchanged.

## Reproduction

Run `tests/benchmark/test_footprint_metrics.py` for analytic geometry, dispatch,
producer, pairing and compatibility regressions. The test's base fallback calls
legacy scalars so failures show the wrong physical values rather than missing
new imports. Passing legacy controls are not counted as base-failing witnesses.

```bash
uv run python scripts/validation/run_empty_world_sweep.py \
  --head-sha HEAD --suite both --seeds 1001 1002 --workers 4 \
  --footprint-metrics --output-dir <external-output>
uv run python scripts/validation/measure_footprint_metrics.py \
  --head-sha HEAD --workers 4 --arms goal orca --output-dir <external-output>
```

Measurement defaults to dev seeds 1001–1030 and the goal arm; an empty `--arms`
selects the full release roster. It uses the canonical campaign runner, all 48
release scenarios and authored horizons. The output is diagnostic-only. The
summary reports paired finite means alongside finite counts; lost references
and execution failures remain visible in campaign records. Raw evidence stays
outside the worktree and the PR reports its measured coverage explicitly.
