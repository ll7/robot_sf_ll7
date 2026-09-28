<!-- AI-GENERATED (robot_sf#9735) - NEEDS-REVIEW -->
# VV-5 fault-injection diagnostic (#9735)

Fixture-level checker sensitivity only; no nominal release or planner-behaviour claim.

Detection: 3/4 injected fixtures. Exact checker and map SHA-256 values appear below.

| Fault | Spawn-validity check | Release-row anomalies |
| --- | --- | --- |
| `reset_pedestrian_overlap` | detected | not_applicable |
| `respawn_onto_robot` | detected | not_applicable |
| `robot_start_inside_wall_radius` | detected | not_applicable |
| `collision_total_doubled` | not_applicable | missed |

## Misses and coverage boundary

- `collision_total_doubled`: Recompute total_collision_count and collisions from typed component counts and the event ledger; block any mismatch before release aggregation.
- The collision-total miss is tracked as release-blocking issue #9855.
- A separate untested reset-overlap case with `route_complete=True` can pass spawn validity; release-blocking issue #9861 owns that outcome-independent gate. The exercised reset-overlap fixture uses `route_complete=False`.
- Unavailable reset clearance can also leave `invalid_run=False`; #9861 owns fail-closed handling for unmeasured starts. This packet did not inject that error.
- `respawn_onto_robot` supplies a synthetic respawn event and matching collision event to `build_spawn_validity`; it tests attribution, not simulator respawn placement.
- The six remaining issue faults were not injected in this packet: `per_cell_wall_force_sum`, `centre_distance_threshold_and_wrong_braking`, `infeasible_gap`, `planner_only_radius_halved`, `planner_goal_direction_flipped`, `pedestrian_robot_force_disabled`.

The controls and mutants use the production checker functions on synthetic inputs. This does not prove scenario-matrix coverage, runtime wiring, or causal attribution.

## Source digests

- `scripts/validation/run_issue_9735_fault_injection.py`: `80c267edfe642fc193552092ec2adc6d66727c5e9d617ecdd24dce78fec5ebc4`
- `robot_sf/sim/spawn_validation.py`: `fe211644ab9e7f80d65e4c97230c0d880c1d6ed62db6588103ac52d9deedd13b`
- `robot_sf/benchmark/spawn_validity.py`: `14974ade8ff94b0422230af910d43da9e66a99a123b894f913a71d4b512f1a77`
- `robot_sf/analysis_workbench/release_row_anomalies.py`: `049ed97cd5c388cc743bfbb130c1c9dcb2c73752adaa8b699613849b0f9e2759`
- `maps/svg_maps/classic_head_on_corridor.svg`: `89f439ac1f04561e0cd550c16be01a9674dc6fe80ba196f0dcc8f8f3c641f11b`
