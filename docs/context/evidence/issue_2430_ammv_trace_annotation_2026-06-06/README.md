# Issue #2430 AMMV Trace Annotation Summary (2026-06-06)

Related issues: [#2430](https://github.com/ll7/robot_sf_ll7/issues/2430),
[#2159](https://github.com/ll7/robot_sf_ll7/issues/2159),
[#2281](https://github.com/ll7/robot_sf_ll7/issues/2281),
[#2428](https://github.com/ll7/robot_sf_ll7/issues/2428)

This bundle records a compact frame-level annotation decision for the Issue #2428
`default_social_force` versus `ammv_social_force` trace pair.

Claim boundary: diagnostic-only telemetry parity check. The selected trace pair is useful for
proving that AMMV force-vector telemetry, selected actions, robot state, and pedestrian state are
preserved in promoted `simulation_trace_export.v1` traces. It is not useful as AMMV
behavioral-difference evidence because the two frame streams are numerically identical over the
promoted 20-frame window.

## Contents

- `summary.json`: exact-frame parity result, annotation anchors, missing richer-diagnostic fields,
  and validation notes.

## Result

The promoted traces both cover `classic_head_on_corridor_low`, seed `111`, steps `0..19`, and
`time_s` `0.1..2.0`. The only differences are top-level source metadata such as `planner_id`,
`episode_id`, and `generated_by`; all recorded per-frame robot, pedestrian, selected-action, event,
and `ammv.pedestrian_force_vectors` fields match exactly.

The recommended Issue #2159 decision is therefore: count the Issue #2428 pair as a telemetry and
rendering proof, not as a mechanism-difference trace-review case. If the AMMV lane needs behavioral
contrast, select another row or seed before adding more annotation infrastructure.
