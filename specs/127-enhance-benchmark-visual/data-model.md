# Data Model: Benchmark Visual Artifacts Enhancement

Purpose: Define internal entities (non-breaking to existing benchmark episode schema) used for SimulationView video generation, plotting manifest, and performance instrumentation.

## Entities

### ReplayState (internal)
| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| episode_id | str | non-empty | Links to benchmark episode record id |
| positions | list[tuple[float,float]] | len>=2; len==len(headings)==len(timestamps) | Robot (x,y) world coordinates per timestep |
| headings | list[float] | radians or degrees (assume radians) | Orientation; used to draw heading arrow (future) |
| timestamps | list[float] | strictly increasing; start>=0 | Simulation relative time seconds |
| ped_positions | list[list[tuple[float,float]]] | optional; len==len(timestamps) | Each frame's pedestrian positions; may be None if unavailable |

Validation failures → adapter marks episode insufficient for video (skip reason: `insufficient-replay-state`).

### VideoArtifact Manifest Entry
| Field | Type | Required | Enum / Constraints |
|-------|------|----------|---------------------|
| episode_id | str | yes | non-empty |
| filename | str|null | conditional | Present if status==success |
| renderer | str | yes | `simulation_view` or `synthetic` |
| status | str | yes | `success`, `skipped`, `failed` |
| note | str|null | optional | One of canonical skip/fail notes or null |
| encode_time_s | float | optional | Present when status==success |
| memory_peak_mb | float | optional | Present when measured (SimulationView path) |

### PlotArtifact Manifest Entry
| Field | Type | Required | Notes |
| name | str | yes | Unique logical plot id |
| filename | str|null | conditional | Written if success |
| status | str | yes | success|skipped|failed |
| note | str|null | optional | Skip/fail note |

### PerformanceVisuals
| Field | Type | Required | Notes |
| plots_time_s | float | yes | Total plots wall time |
| first_video_time_s | float|null | optional | Null if no video attempted |
| plots_over_budget | bool | yes | plots_time_s > 2.0 |
| video_over_budget | bool | yes | first_video_time_s > 5.0 (if not null) |
| memory_over_budget | bool | optional | Peak memory > 100 MB (SimulationView path) |

## Canonical Skip / Note Codes
```
SMOKE_MODE = "smoke-mode"
DISABLED = "disabled"
SIM_VIEW_MISSING = "simulation-view-missing"
MOVIEPY_MISSING = "moviepy-missing"
INSUFFICIENT_REPLAY = "insufficient-replay-state"
```
Future failure code pattern: hyphenated lower-case tokens.

## Relationships
- One `VideoArtifact` references exactly one `ReplayState` (if status success or failed during encode). Skipped artifacts may lack a ReplayState object (insufficient state or global skip reason).
- `PerformanceVisuals` aggregates timing across plots and first video only.
- `PlotArtifact` independent of ReplayState.

## Invariants
- For each successful video artifact, `filename` exists and is reproducible `video_<episode_id>.mp4`.
- For synthetic fallback path, `renderer==synthetic` and no memory metrics recorded.
- For SimulationView path with success, memory metrics recorded when psutil available (optional otherwise).

## Non-Goals
- Do not expose ReplayState externally yet (internal helper, subject to change without schema version bump).
- Do not embed full pedestrian force data or low-level physics; out of current FR scope.

## Open Considerations (Documented for future)
- Potential addition of ped trajectories to enhance visualization richness.
- Adaptive FPS keyed to simulation timestep (would require storing dt in ReplayState).
