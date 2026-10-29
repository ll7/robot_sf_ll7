# Predictive pedestrian TTC reward

`time_to_collision` is the minimum time in seconds to constant-velocity robot–pedestrian
**disc contact**, using physical robot world velocity, backend pedestrian velocities,
and robot/pedestrian footprint radii. Existing overlap reports zero; no future contact reports
infinity. Missing pedestrian velocities cannot establish future contact.

Holonomic drives supply `state.velocity_xy` directly in both command modes, preserving
reverse direction even when `current_speed` reports a nonnegative magnitude. Differential
and bicycle drives project their signed forward speed along the current pose heading.

The shaped `ttc_risk` term is `min(1, 1 / max(TTC, 0.001))`; zero TTC is maximum risk.
Missing/non-finite TTC gives zero risk, never a duplicate of `near_miss`.
The clearance term continues to report its independent per-step proximity event.

This changes reward shaping for future training. It does not change benchmark evaluation
of existing checkpoints: observations, actions, termination and canonical benchmark metrics
are unchanged. No training or benchmark campaign was run for this repair.
