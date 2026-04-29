# Failure Taxonomy

The policy-search reporting layer uses one dominant failure mode per failed
episode. The classification is heuristic and intentionally limited to fields
that already exist in benchmark episode JSONL records.

## Modes

- `static_collision`: collision in a scenario with no configured pedestrians.
- `pedestrian_collision`: collision when pedestrian agents are present.
- `near_miss_intrusive`: no terminal collision, but a near-miss metric or
  distance-band violation is present.
- `deadlock`: max-step or termination outcome with essentially no progress.
- `oscillation`: repeated command-sign changes when that metric is exposed.
- `timeout_low_progress`: timeout without a more specific failure signal.
- `wrong_waypoint_behavior`: explicit runtime error or malformed control path.
- `bottleneck_yield_failure`: low-speed timeout in doorway or bottleneck cases.
- `overconservative_stop`: low-speed timeout outside the bottleneck-specific case.

## Current Limitation

The taxonomy is episode-record-driven. It does not yet distinguish all static
versus dynamic collision subtypes that would require per-step replay features.