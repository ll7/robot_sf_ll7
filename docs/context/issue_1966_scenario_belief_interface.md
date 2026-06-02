# Issue #1966 ScenarioBelief Interface Design

Status: design proposal / diagnostic evidence, June 1, 2026.

Related issue: [Issue #1966](https://github.com/ll7/robot_sf_ll7/issues/1966)

## Summary

Robot SF should introduce a sensor-agnostic `ScenarioBelief` layer between
sensor/perception inputs and policy-observation projections. The layer is a semantic,
uncertainty-aware scenario representation: sensor-specific code terminates at input adapters, and
learned-policy or planner-facing observations are deterministic projections from the belief.

This note is design-only. It does not replace current observation modes, retrain policies, certify
real sensors, or create benchmark-performance evidence.

## Current Contract Surfaces

Robot SF already has several observation and projection surfaces:

- default Gym observations in `docs/dev/observation_contract.md`: `drive_state` plus stacked
  LiDAR-style `rays`;
- `SOCNAV_STRUCT` observations from `robot_sf/sensor/socnav_observation.py`, with robot, goal,
  pedestrian, map, simulation, and optional predictive fields;
- occupancy-grid augmentation through `robot_sf/nav/occupancy_grid.py`, including fixed channel
  order and metadata fields;
- graph-style projection from SocNav fields in `robot_sf/sensor/social_graph_observation.py`;
- benchmark observation-level metadata in `robot_sf/benchmark/observation_levels.py`, which records
  perception assumptions such as `oracle_full_state`, `tracked_agents_no_noise`, `lidar_2d`, and
  `occluded_partial_state`.

These surfaces are policy or benchmark contracts, not a unified semantic belief contract. In
particular, the graph adapter currently converts an existing SocNav observation into graph tensors;
it is not an input-source-neutral scene representation.

Two existing context notes define adjacent boundaries that this design should preserve:

- `docs/context/issue_1246_observation_levels.md` defines observation levels as benchmark evidence
  labels and compatibility gates, not sim-to-real validity claims.
- `docs/context/issue_1612_observation_track_architecture.md` defines benchmark observation tracks
  as aggregation fences. Rows from different observation contracts must not be merged as one
  result set unless a report explicitly frames them as cross-track diagnostics.

## Decoupling Principle

Supported sensor or perception inputs should build the same `ScenarioBelief` schema. Differences in
sensor quality should appear as uncertainty, confidence, visibility, source, and stale-data fields,
not as unrelated policy-facing semantic contracts.

```text
simulator oracle / lidar rays / tracked agents / occupancy grid / detections
  -> input adapter
  -> ScenarioBeliefContract
  -> policy-observation projection
  -> planner or learned local policy
```

The policy-facing interface is a projection of `ScenarioBelief`. It may remain a fixed-size tensor,
flattened Gymnasium `Dict`, occupancy-grid tensor, graph bundle, or legacy `SOCNAV_STRUCT`
projection.

## Contract Split

Use two separate contracts:

- `ScenarioBeliefContract`: semantic, uncertainty-aware, human-readable, adapter-facing.
- `PolicyObservationContract`: tensor, graph, dict, or raster projection consumed by a planner or
  learned local policy.

This split avoids treating a heterogeneous, variable-length belief object as if every RL backend
could consume it directly. Stable-Baselines3-compatible policies still need bounded spaces,
padding, masks, and flattened keys. Graph-capable policies can consume variable entity/edge counts
through a graph projection, but that is still a projection contract.

## Proposed Minimal Schema

The first Robot SF contract should be a typed, JSON-serializable dataclass family. A code follow-up
can place it under `robot_sf/representation/scenario_belief.py`.

```python
@dataclass(frozen=True)
class ScenarioBelief:
    schema_version: str
    frame_id: str
    sim_time_s: float
    timestep_s: float
    scenario_id: str | None
    seed: int | None
    map: MapBelief
    ego: EntityBelief
    goals: tuple[GoalBelief, ...]
    agents: tuple[EntityBelief, ...]
    obstacles: tuple[ObstacleBelief, ...]
    relations: tuple[SceneRelationBelief, ...]
    observation_assumptions: ObservationAssumptions
    source_summary: BeliefSourceSummary
```

Required common field semantics:

- values are estimates, not necessarily ground truth;
- all positions declare frame and units;
- fixed-size policy projections must define padding, masks, normalization, and clipping;
- optional fields must distinguish unknown from missing-by-design;
- every inferred property carries source and time validity when the input path can supply it.

## Entity Beliefs

`EntityBelief` should represent the ego robot, pedestrians, dynamic obstacles, static obstacles
when modeled as entities, goals when useful for relations, and unknown objects.

Required fields:

- stable `entity_id`;
- `type_distribution`: `ego_robot`, `pedestrian`, `static_obstacle`, `dynamic_obstacle`, `goal`,
  `boundary`, `unknown`;
- pose estimate, frame, and pose uncertainty;
- velocity estimate and uncertainty when available;
- radius or footprint estimate and uncertainty;
- heading estimate and uncertainty when meaningful;
- existence probability and class confidence;
- source timestamp, last-observed age, and optional prediction horizon;
- visibility state;
- debug label.

Simulator oracle state is represented as a high-confidence source, not as a separate policy
interface.

## Map And Obstacle Beliefs

`MapBelief` and `ObstacleBelief` should cover the existing map geometry and occupancy-grid
surfaces without forcing all policies to consume raster channels.

Obstacle/map fields:

- geometry estimate: polygon, segment, circle, grid cell set, or unknown footprint;
- occupancy probability, free-space probability, and unknown probability when rasterized;
- semantic class distribution such as wall, curb, boundary, temporary obstacle, or unknown obstacle;
- resolution and coordinate frame metadata;
- source and update timestamp;
- optional relation to occupancy-grid channels.

The existing occupancy-grid projection should remain a policy projection with values in `[0, 1]`,
the configured channel order from `OBSERVATION_CHANNEL_ORDER`, and metadata fields matching
`OccupancyGrid.metadata_observation()`.

## Relations

`SceneRelationBelief` should make derived social-navigation quantities explicit instead of hiding
them in one projection.

Candidate relation types:

- `near`;
- `visible`;
- `occluded_by`;
- `blocks_route`;
- `social_force`;
- `time_to_collision`;
- `goal_progress`;
- `inside_fov`;
- `unknown_interaction`.

Each relation should include source id, target id, scalar features, confidence or uncertainty, and
provenance. Social-force terms are relations or derived diagnostics, not mandatory entity fields.

## Unknown And Partial Observability

The representation must not encode absence of evidence as evidence of absence.

Use first-class states:

- `free`;
- `occupied`;
- `unknown`;
- `unobserved`;
- `occluded`;
- `out_of_range`;
- `outside_fov`;
- `stale_prediction`;
- `low_confidence_class`;
- `low_confidence_velocity`.

The existing dynamic occlusion helper in `robot_sf/sensor/socnav_observation.py` can inform the
first `visible` / `occluded_by` relation vocabulary, but the belief contract should not mutate
ground-truth simulator state.

## Input Adapters

Input adapters are the only sensor-specific layer.

| Adapter | Input | ScenarioBelief output boundary |
| --- | --- | --- |
| `OracleAdapter` | simulator state | Low-uncertainty robot, goal, pedestrian, obstacle, and map beliefs. |
| `TrackedAgentAdapter` | tracked pedestrians/objects | Entity beliefs with covariance, track age, class confidence, and stale-track state. |
| `LidarAdapter` | range rays | Free/occupied/unknown spatial beliefs; weak or absent semantic labels. |
| `OccupancyGridAdapter` | occupancy-grid channels | Occupancy/free/unknown map and obstacle hypotheses with grid resolution/source metadata. |
| `CameraDetectionAdapter` | detections | Class-probabilistic entity beliefs; depth and velocity are missing or low-confidence unless tracked. |
| `FusionAdapter` | multiple sources | Fused belief with source provenance and conflict handling. |

Use "supported sensor or perception input" rather than "all possible sensors". The schema can be
shared even when field completeness and calibration quality differ.

## Policy Projections

Policy projections consume `ScenarioBelief` and emit the existing or future policy contract:

| Projection | Output | Compatibility rule |
| --- | --- | --- |
| `to_socnav_struct()` | Existing nested/flattened SocNav keys | Preserve current key names and frame semantics unless an explicit migration is opened. |
| `to_fixed_tensor()` | SB3-compatible tensor or dict | Include masks, padding, normalization, and missing-value policy. |
| `to_occupancy_grid()` | Grid tensor plus metadata | Preserve channel order and `[0, 1]` value contract. |
| `to_lidar_rays()` | Range vector/stack | Preserve `rays` shape and normalization from the default Gym contract. |
| `to_graph_instance()` | Graph-style robot/entity/edge tensors | Keep current graph adapter deployment safety: no future or label-only fields. |
| `to_debug_json()` | Compact JSON/YAML | Include uncertainty/source/visibility fields for human diffing. |

Projection code should be deterministic. If a projection drops uncertainty or source fields, it
must record that reduction in diagnostics or metadata.

Benchmark projection metadata should continue to carry `observation_level`, `observation_mode`, and
track fields where applicable. `ScenarioBelief` can reduce adapter drift, but it does not make
different benchmark tracks directly comparable by itself.

## Frame And Normalization Boundaries

The belief contract should record raw semantic values in declared frames and units. Normalization
belongs to projections.

Required boundaries:

- world-frame versus ego-frame positions and velocities are explicit;
- `SOCNAV_STRUCT` preserves current world-frame robot/goal positions and robot-frame pedestrian
  velocities;
- occupancy grids preserve grid origin, resolution, size, center-on-robot flag, ego-frame flag, and
  channel indices;
- graph projections declare robot-frame rotation and distance sorting;
- fixed-size projections declare caps, padding values, and masks;
- debug JSON/YAML should prefer readable physical units over normalized tensors.

## Representation-Quality Proxies

These checks are diagnostic only. They are not benchmark success, SNQI, or paper-facing
performance claims.

- Input equivalence: oracle, tracked-agent, ray, and grid inputs produce the same belief schema with
  different source/uncertainty fields.
- Projection consistency: `SOCNAV_STRUCT` keys can be reconstructed from sufficient-confidence
  beliefs.
- Grid consistency: occupancy-grid channels describe the same obstacles, pedestrians, and robot as
  the belief where the input can support that mapping.
- Uncertainty propagation: covariance, confidence, existence probability, visibility, and
  stale-track age are preserved or explicitly reduced.
- Visibility consistency: FOV, static occlusion, and dynamic occlusion produce explicit visibility
  or `occluded_by` metadata.
- Scenario coverage: representative scenarios report entity counts, relation counts, and missing
  fields.
- Debug readability: compact JSON/YAML diffs explain why two beliefs or projections differ.
- Policy-contract stability: existing observation keys remain unchanged unless a migration is
  deliberate.

## MVP Decoupling Test

The first implementation proof should test decoupling, not only serialization.

MVP claim:

```text
Two input paths produce the same ScenarioBelief schema and the same policy-observation schema,
differing only in uncertainty, source, visibility, and missing-data fields.
```

Input path A:

```text
simulator oracle -> ScenarioBelief -> SOCNAV_STRUCT-compatible projection
```

Input path B:

```text
visibility-limited synthetic perception or lidar/ray input
  -> ScenarioBelief
  -> the same SOCNAV_STRUCT-compatible projection plus masks/uncertainty diagnostics
```

Both paths should:

- validate against the same `ScenarioBelief` schema;
- project to the same policy-observation key set;
- expose different confidence, uncertainty, visibility, source, and stale-track values;
- preserve backward compatibility with existing observation keys;
- emit debug JSON/YAML explaining the difference between oracle and partial-observation beliefs.

## Implementation Slice

Recommended follow-up structure:

```text
robot_sf/representation/
  scenario_belief.py
  input_adapters/
    simulator_oracle.py
    socnav_struct.py
    lidar_rays.py
  projections/
    socnav_struct.py
    debug_json.py
    graph.py
tests/representation/
  test_scenario_belief_socnav_projection.py
  test_scenario_belief_debug_json.py
  test_scenario_belief_uncertainty_fields.py
  test_scenario_belief_decoupling.py
```

First implementation target:

- construct `ScenarioBelief` from one simulator step as high-confidence oracle input;
- construct `ScenarioBelief` from one partial-observation path;
- project both to the same SocNav structured key set;
- emit compact debug JSON with uncertainty/source fields;
- assert no drift in existing observation keys.

### Issue #2105 MVP Slice 2026-06-02

Issue #2105 implements the first narrow code proof for the MVP decoupling test. The slice adds
`robot_sf/representation/scenario_belief.py` with a typed `ScenarioBelief` dataclass contract,
simulator-oracle and visibility-limited simulator adapters, a `SOCNAV_STRUCT`-compatible policy
projection, and deterministic debug metadata.

Claim boundary: this is representation and compatibility evidence only. It does not claim benchmark
improvement, SNQI movement, calibrated real-sensor uncertainty, or paper-facing performance.

Validation recorded for the MVP slice:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/representation -q
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/sensor tests/test_socnav_observation.py tests/test_socnav_observation_mode.py tests/test_sensor_fusion_stack.py tests/test_range_sensor.py -q
scripts/dev/run_worktree_shared_venv.sh -- uv run ruff check robot_sf/representation tests/representation
scripts/dev/run_worktree_shared_venv.sh -- uv run ruff format --check robot_sf/representation tests/representation
git diff --check
```

## Follow-Up Risks

- Scope risk: do not imply all real sensors are supported; start with Robot SF's supported synthetic
  and structured inputs.
- Compatibility risk: changing observation keys can break trained policies; keep projection tests
  strict.
- Evidence risk: representation-quality proxies can be mistaken for benchmark evidence; keep them
  diagnostic.
- Calibration risk: synthetic uncertainty is not real sensor calibration.
- Complexity risk: graph, grid, ray, image, and fused projections can expand quickly; keep the first
  implementation slice narrow.

## Design-Only Validation

Use cheap validation for this design-only note:

```bash
test -f docs/dev/observation_contract.md
test -f docs/training/environment_contract.md
test -f robot_sf/sensor/socnav_observation.py
test -f robot_sf/nav/occupancy_grid.py
test -f robot_sf/benchmark/observation_levels.py
test -f robot_sf/sensor/social_graph_observation.py
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
