# Issue #1612 Observation-Track Benchmark Architecture

Date: 2026-05-30

## Goal

Issue #1612 defines how Robot SF should run parallel benchmark tracks with different planner
observation contracts without mixing incompatible evidence. The immediate split is between the
existing grid/SocNav-state benchmark line and the LiDAR-observation line started by #1613, but the
architecture should also leave room for future privileged, noisy-tracking, occlusion, or adapter
tracks.

This is a design note only. It does not launch a benchmark campaign, train policies, or promote
LiDAR results as comparable with grid-observation results.

## Current Surfaces

The design builds on existing additive metadata rather than replacing the benchmark runner:

- `docs/benchmark_spec.md` already documents `--observation-mode` and the supported observation
  mode vocabulary for runner invocations.
- `docs/dev/observation_contract.md` defines environment observation modes and benchmark
  observation levels.
- `robot_sf/benchmark/observation_levels.py`, `robot_sf/benchmark/algorithm_metadata.py`, and
  `robot_sf/benchmark/planner_command_contract.py` validate planner/observation combinations.
- `robot_sf/benchmark/map_runner.py` records `observation_mode` and `observation_level` at the
  top level, in `scenario_params`, and inside `algorithm_metadata`.
- `configs/benchmarks/lidar/observation_track_smoke_issue_1613.yaml` is the first concrete
  LiDAR-track smoke packet. It reuses `configs/scenarios/sanity_v1.yaml`, declares
  `benchmark_observation_level=lidar_2d` and `active_observation_mode=sensor_fusion_state` under
  `observation_track`, and mirrors `observation_level=lidar_2d` plus
  `observation_mode=sensor_fusion_state` in the planner row.

## Vocabulary

Use the word **track** for a benchmark evidence lane with a stable observation contract and
aggregation boundary. A track is not just a planner family and not just an environment observation
mode.

Required track fields:

| Field | Required value shape | Purpose |
| --- | --- | --- |
| `benchmark_track` | stable slug, for example `grid_socnav_v1` or `lidar_2d_v1` | Primary aggregation fence. Rows with different values must not be aggregated as one result set. |
| `observation_level` | vocabulary from `robot_sf/benchmark/observation_levels.py` | Coarse perception assumption, such as `oracle_full_state` or `lidar_2d`. |
| `observation_mode` | runner/environment mode, such as `socnav_state` or `sensor_fusion_state` | Active observation producer and policy input mode. |
| `runtime_inputs` | list of semantic inputs | Human-readable guard against hidden privileged inputs. |
| `privileged_runtime_inputs` | boolean | Explicitly distinguishes benchmark tracks that may use simulator state. |
| `adapter_mode` | `native`, `native_or_mixed_policy_action`, `adapter`, `perception_adapter`, `diagnostic_stub`, `fallback`, or `degraded` | Separates native rows from adapter-derived or caveated execution. |
| `planner_mode` | `classical`, `learned_checkpoint`, `learned_stub`, `hybrid`, or `external_wrapper` | Makes planner-family grouping independent of observation-track grouping. |
| `track_schema_version` | semver-like slug, for example `observation-track.v1` | Lets future schema additions remain additive. |

Recommended optional fields:

- `environment_observation_keys`: concrete environment observation keys such as `drive_state`, `rays`, or
  `occupancy_grid`.
- `sensor_geometry`: ray count, range, field of view, stack steps, or grid resolution when those
  values define the track.
- `goal_encoding`: how route/goal information enters the observation.
- `excluded_runtime_inputs`: explicit negative contract for simulator state, future trajectories,
  labels, or map occupancy.
- `model_registry_id`: required when the row evaluates a learned checkpoint.
- `track_claim_boundary`: one line stating whether the row is benchmark evidence, diagnostic-only,
  or contract smoke.

## Canonical Tracks

| Track | `benchmark_track` | `observation_level` | Typical `observation_mode` | Runtime inputs | Evidence boundary |
| --- | --- | --- | --- | --- | --- |
| Grid/SocNav state | `grid_socnav_v1` | `tracked_agents_no_noise` or `oracle_full_state` when explicitly privileged | `socnav_state` or PPO dict/grid contracts | robot state, goal, tracked pedestrian state, optional occupancy grid | Current map-based benchmark line. Do not aggregate with LiDAR rows unless a report explicitly asks for cross-track comparison. |
| LiDAR 2D | `lidar_2d_v1` | `lidar_2d` | `sensor_fusion_state` | `drive_state`, `rays`, goal encoded in drive state | Evidence only when rows carry LiDAR metadata and use a compatible policy/adapter. #1613 is contract smoke, not performance evidence. |
| Privileged diagnostic | `privileged_state_v1` | `oracle_full_state` | runner-specific | simulator state, full map, labels only when declared | Diagnostic or upper-bound evidence. Must not be mixed into baseline-safe performance claims. |
| Adapter-derived perception | `adapter_perception_v1` | source level plus adapter caveat | adapter-specific | native planner input plus transformed perception features | Valid only with `adapter_mode=perception_adapter` and adapter provenance. |

## Config Layout

Prefer additive track metadata near benchmark configs instead of duplicating scenario matrices:

```yaml
schema_version: observation-track-benchmark.v1
benchmark_track: lidar_2d_v1
track_schema_version: observation-track.v1
scenario_matrix: configs/scenarios/sanity_v1.yaml
scenario_reuse_policy:
  source: configs/scenarios/sanity_v1.yaml
  rationale: Reuse scenario geometry while changing only the observation contract.
observation_track:
  benchmark_observation_level: lidar_2d
  active_observation_mode: sensor_fusion_state
  runtime_inputs:
    - robot_state
    - goal
    - lidar_rays
  environment_observation_keys:
    - drive_state
    - rays
  privileged_runtime_inputs: false
  excluded_runtime_inputs:
    - occupancy_grid
    - simulator_backed_global_map
    - future_pedestrian_trajectories
    - collision_or_success_labels
```

Naming convention:

- Put first-class LiDAR benchmark configs under `configs/benchmarks/lidar/`.
- Keep grid/SocNav benchmark configs under the existing benchmark config roots, but add
  `benchmark_track: grid_socnav_v1` when a config is being touched for track-aware work.
- Use `*_smoke_issue_<issue>.yaml` for non-performance contract smokes.
- Use `*_pilot_v<N>.yaml` for bounded benchmark pilots.
- Use `*_release_v<N>.yaml` only after track metadata is complete and validation has passed.

## Result Schema Boundary

After Issue #1702, every track-aware episode row should expose the same track metadata in three
places:

- top-level fields for direct JSONL/Parquet filtering,
- `scenario_params` for resume identity and run provenance,
- `algorithm_metadata.observation_level`, `algorithm_metadata.observation_spec`, and a new
  `algorithm_metadata.benchmark_track` block for planner-contract provenance.

Minimum row fields:

```json
{
  "benchmark_track": "lidar_2d_v1",
  "track_schema_version": "observation-track.v1",
  "observation_level": "lidar_2d",
  "observation_mode": "sensor_fusion_state",
  "planner_mode": "learned_checkpoint",
  "adapter_mode": "native",
  "privileged_runtime_inputs": false
}
```

Aggregators and reports should group by `benchmark_track` before planner or scenario unless the
report is explicitly a cross-track diagnostic. Cross-track diagnostics must label rows as
non-comparable when observation contracts differ.

## Shared And Track-Specific Surfaces

Shared across tracks:

- scenario matrices and map definitions when the geometry semantics match,
- seed-set files and seed policy,
- core episode schema, metric functions, fallback/degraded row policy, and artifact provenance,
- planner readiness metadata when the planner supports the requested observation contract.

Track-specific:

- observation metadata blocks in benchmark configs,
- learned checkpoint registry fields and eligibility checks,
- adapter/perception transforms,
- report sections and aggregation defaults,
- validation smokes that assert the active policy received only the allowed runtime keys.

Do not fork scenario definitions just to create an observation track. Fork or derive scenario
definitions only when the scenario semantics themselves change.

## Reporting Rules

- Report `benchmark_track` in every table header or row group.
- Do not average across `grid_socnav_v1` and `lidar_2d_v1` as if they were one benchmark.
- When a report compares tracks, use paired language: "same scenario/seed under different
  observation contracts", not "planner A is better than planner B" unless the planner and
  observation contract match.
- Treat `fallback`, `degraded`, and `diagnostic_stub` rows as limitations or contract smoke, not
  benchmark success evidence.
- If a learned policy is involved, include the model registry observation-contract tags in the
  report or declare the row ineligible for benchmark claims.

## Example Labeling

Existing grid smoke:

- Config: `configs/benchmarks/observation_mode_goal_parity_smoke.yaml`
- Proposed track metadata: `benchmark_track=grid_socnav_v1`,
  `observation_level=tracked_agents_no_noise`, and `observation_mode=goal_state` or
  `socnav_state` per planner entry.
- Interpretation: input-contract parity smoke for the built-in `goal` planner, not a LiDAR row.

LiDAR smoke:

- Config: `configs/benchmarks/lidar/observation_track_smoke_issue_1613.yaml`
- Track metadata: `benchmark_track=lidar_2d_v1`, `observation_level=lidar_2d`,
  `observation_mode=sensor_fusion_state`, `observation_keys=[drive_state, rays]`.
- Interpretation: contract smoke with stubbed/compatible policy path. No performance claim until a
  durable LiDAR checkpoint or explicit diagnostic adapter is available.

## Follow-Up Implementation Issues

The design should be implemented incrementally:

1. Issue #1702: add first-class `benchmark_track` metadata support to benchmark config loading,
   episode rows, resume identity, and algorithm metadata.
2. Issue #1703: add report/aggregation guards that warn or fail when incompatible
   `benchmark_track` values are aggregated without an explicit cross-track mode.
3. Issue #1704: add model-registry observation-track fields for learned checkpoints and extend the
   eligibility checker to require them before benchmark promotion.

## Validation

Design validation for this note:

- Reviewed `docs/benchmark_spec.md`, `docs/benchmark.md`,
  `docs/benchmark_planner_family_coverage.md`, and `docs/dev/observation_contract.md`.
- Reviewed `configs/benchmarks/observation_mode_goal_parity_smoke.yaml` as the existing grid/state
  smoke example.
- Reviewed `configs/benchmarks/lidar/observation_track_smoke_issue_1613.yaml` and
  `tests/benchmark/test_lidar_observation_track.py` as the LiDAR-track coexistence example.

No benchmark result is claimed here. The proof is architectural consistency with current config and
metadata surfaces.
