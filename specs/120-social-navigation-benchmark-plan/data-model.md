# Data Model — Social Navigation Benchmark

Version: 0.1 (Design Phase)  
Schema Version Reference: `episode.schema.v1`

## Entity Overview
- ScenarioSpec
- ScenarioMatrix
- EpisodeRecord
- MetricsBundle
- SNQIWeights
- AggregateSummary
- ResumeManifest

## 1. ScenarioSpec
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | string | yes | Unique scenario identifier (kebab-case). |
| density | float | yes | Nominal average pedestrian per m^2 (target). |
| flow_pattern | string | yes | Enum: straight, bidirectional, crossing, maze, wave, group. |
| obstacles | string | yes | Enum: none, simple, bottleneck, maze_sparse, maze_dense. |
| group_behavior_flags | list[string] | no | Flags: group_cohesion, adversarial, mixed. |
| repetitions | int | yes | Number of episodes to run for this scenario. |
| map_id | string | yes | Reference to map asset or layout preset. |
| seed_offset | int | no | Base offset added to global seed per repetition. |

## 2. ScenarioMatrix
| Field | Type | Required | Description |
| scenarios | list[ScenarioSpec] | yes | Collection of scenario specs. |
| schema_version | string | yes | Scenario matrix schema tag. |
| created_at | string (ISO8601) | yes | Timestamp of matrix creation. |

## 3. EpisodeRecord
| Field | Type | Required | Description |
| episode_id | string | yes | Deterministic unique id (hash). |
| schema_version | string | yes | Episode schema tag. |
| scenario_id | string | yes | Link to ScenarioSpec.id. |
| scenario_params | object | yes | Canonical parameters subset (density, flow, etc.). |
| repetition_index | int | yes | 0-based index within scenario repetitions. |
| seed | int | yes | Global seed used. |
| algo_id | string | yes | Planner/algorithm identifier. |
| metrics | MetricsBundle | yes | Computed metrics subtree. |
| status | string | yes | Enum: success, collision, timeout, aborted. |
| timings | object | yes | { wall_clock_s: float, steps: int }. |
| provenance | object | yes | { git_sha, config_hash, scenario_hash, weights_version? }. |
| created_at | string | yes | ISO8601 generation time. |

## 4. MetricsBundle (Nested)
| Field | Type | Description |
|-------|------|-------------|
| success | int (0/1) | Episode goal reached status. |
| time_to_goal | float | Seconds or simulation time steps scaled. |
| path_efficiency | float | Ratio: shortest_path_length / actual_path_length. |
| collisions | int | Collision count. |
| near_misses | int | Count of distance<threshold_near_miss without collision. |
| min_interpersonal_distance | float | Minimum distance encountered. |
| mean_interpersonal_distance | float | Arithmetic mean distance. |
| force_mean | float | Mean norm of interaction force on robot. |
| force_p95 | float | 95th percentile force norm. |
| force_exceedance_events | int | Number of timesteps force>comfort threshold. |
| comfort_exposure | float | Proportion timesteps force>threshold. |
| smoothness_jerk_mean | float | Mean jerk magnitude. |
| smoothness_curvature_mean | float | Mean path curvature. |
| energy_accel_sum | float | Sum |acceleration| over episode. |
| gradient_norm_mean | float | Mean gradient norm along trajectory (if available). |
| snqi | float | Composite index value (if weights provided). |

## 5. SNQIWeights
| Field | Type | Required | Description |
| weights_version | string | yes | Version tag (semantic). |
| created_at | string | yes | Timestamp. |
| git_sha | string | yes | Source commit. |
| baseline_stats_path | string | yes | Path to stats file used. |
| baseline_stats_hash | string | yes | Hash of stats content. |
| normalization_strategy | string | yes | e.g., median_p95_baseline. |
| bootstrap_params | object | no | { samples:int, confidence:float, seed?:int }. |
| components | list[string] | yes | Metric component names in order. |
| weights | object | yes | Mapping metric->float weight. |

## 6. AggregateSummary
| Field | Type | Description |
| group_key | string | Algorithm/scenario grouping id. |
| metrics | object | Map metric_name -> { mean, median, p95, mean_ci?, median_ci?, p95_ci? }. |
| bootstrap | object | { samples:int, confidence:float } if performed. |
| generated_at | string | ISO8601 timestamp. |

## 7. ResumeManifest
| Field | Type | Description |
| schema_version | string | Resume manifest schema tag. |
| episodes_file | string | Path to JSONL episodes file monitored. |
| episodes_count | int | Count of lines indexed. |
| file_size | int | Bytes. |
| mtime | float | Modification time captured. |
| index_hash | string | Hash of concatenated episode_ids for quick invalidation check. |
| updated_at | string | Timestamp. |

## Relationships
- ScenarioMatrix.scenarios[*] → ScenarioSpec
- EpisodeRecord.scenario_id → ScenarioSpec.id
- AggregateSummary.group_key derived from (algo_id or scenario grouping)
- SNQIWeights.weights_version referenced in provenance (optional)
- ResumeManifest index references episodes file lines (implicit relationship)

## Validation Rules
- scenario_id MUST exist in matrix when episode recorded.
- metrics.snqi present only if weights applied.
- path_efficiency in (0, 1] (allow epsilon >1 due to floating rounding but clamp in aggregation).
- collisions >= near_misses? (No; near_misses excludes collisions; treat independently.)
- comfort_exposure in [0,1].
- weights sum not enforced (can be unnormalized) but SNQI computation doc states normalized to 1.0 for interpretable contribution.

## State Transitions (Episode)
Draft:
1. pending (implicit before run)
2. running (not persisted)
3. completed_success / completed_collision / completed_timeout / aborted

Only final terminal state persisted as `status` simplified to { success, collision, timeout, aborted }.

## Open Data Model Questions
- Should we store raw trajectory? (Out of scope first release; large size)
- Include per-step force distribution? (Derived stats only initially)

---
End of data model design.
