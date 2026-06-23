# Benchmark Observation Visibility

Issue: [#1090](https://github.com/ll7/robot_sf_ll7/issues/1090)

Benchmark scenarios can opt into geometric filtering for planner-facing SocNav observations through
an `observation_visibility` block:

```yaml
observation_visibility:
  enabled: true
  fov_degrees: 120.0
  max_range_m: 8.0
  static_occlusion: true
  tracking_noise_std_m: 0.25
```

The filter is applied inside `SocNavObservationFusion` before pedestrians are sorted, truncated, and
returned to benchmark planners. The simulator state, pedestrian buffers, episode metrics, and
recorded ground-truth trajectories remain unchanged. This keeps perception-ablation experiments
auditable: hidden pedestrians are absent from the planner input, but benchmark scoring still uses
the full world state.

## Settings

* `enabled`: turns planner-facing filtering on. If the block is present and `enabled` is omitted,
  filtering is enabled.
* `fov_degrees`: robot-centered horizontal field of view in degrees, in `(0, 360]`.
* `max_range_m`: optional positive range cutoff in meters.
* `static_occlusion`: when true, static map obstacle polygons block line of sight between the robot
  and pedestrian centers.
* `tracking_noise_std_m`: optional deterministic synthetic centroid offset, in meters, applied to
  visible pedestrians in the `ScenarioBelief` visibility-limited adapter. Use `0.0` for exact
  zero-noise reproduction.

The smoke scenario
`configs/scenarios/single/observation_visibility_blind_corner_smoke.yaml` demonstrates all settings
on the Francis 2023 blind-corner map.

## Limits

This is a benchmark-native geometric observation abstraction, not a calibrated sensor simulation.
It does not model probabilistic detection, latency, false positives, pedestrian body extent, or
camera/lidar intrinsics. The `tracking_noise_std_m` knob is a synthetic centroid-offset stressor
for planner-contract tests, not calibrated tracker noise. Static occlusion uses map obstacle
geometry only. Dynamic pedestrian-to-pedestrian occlusion is intentionally deferred to
[#1124](https://github.com/ll7/robot_sf_ll7/issues/1124) so it can be designed with clear
body-shape, ordering, and planner-contract semantics.

Output records include `algorithm_metadata.observation_visibility` for the active settings, and the
scenario identity includes the original `observation_visibility` block for resume and provenance.

## CLEAR Tracking Diagnostics

Issue [#3505](https://github.com/ll7/robot_sf_ll7/issues/3505) adds a diagnostic
bridge for CLEAR Multi-Object Tracking Accuracy (MOTA) and Multi-Object Tracking
Precision (MOTP). The helper compares an oracle `ScenarioBelief` with a
visibility-limited or otherwise degraded `ScenarioBelief` using pedestrian
entity IDs as the correspondence contract.

The metrics remain opt-in and diagnostic. They expose `clear_mota`,
`clear_motp_m`, missed-detection counts, false-positive counts, and ID-switch
counts only when episode metadata carries `clear_tracking_uncertainty.enabled:
true`. Zero-noise oracle-vs-oracle comparison reports perfect MOTA and zero MOTP
offset. These columns do not claim calibrated camera/LiDAR performance or
real-sensor validity.
