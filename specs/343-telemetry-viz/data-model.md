# Data Model: Telemetry Visualization

## Entities

### Telemetry Sample
- **Purpose**: Per-timestep record for live/replay alignment.
- **Fields**:
  - `timestamp` (float, seconds, monotonic)
  - `frame_idx` (int, non-negative, strictly increasing within episode)
  - `episode_id` (string, unique per episode)
  - `seed` (int)
  - `metrics` (object):
    - `fps` (float)
    - `reward` (float)
    - `collisions` (int, cumulative)
    - `min_ped_distance` (float, meters)
    - `action_norm` (float)
  - `status` (enum: running|paused|terminated|truncated)
- **Constraints**: Samples append-only; frame_idx monotonic; status transitions follow Simulation Session below.

### Telemetry Stream
- **Purpose**: Append-only JSONL stream persisted during runs and replayable later.
- **Fields**:
  - `episode_id` (string)
  - `path` (string, artifact location)
  - `schema_version` (string, e.g., `v1`)
  - `decimation` (int, >=1; 1 means full fidelity)
  - `samples` (logical sequence; stored as JSONL lines)
- **Constraints**: Stream metadata stored once; samples may be decimated for UI but full data preserved unless configured otherwise.

### Visualization Session
- **Purpose**: UI/session state for live or replay interaction.
- **Fields**:
  - `mode` (enum: live|replay|headless)
  - `selected_metrics` (list[str])
  - `pane_layout` (enum: vertical_split|horizontal_split)
  - `refresh_hz` (float >= 1.0)
  - `cursor_frame` (int; replay only)
  - `export_targets` (list of Export Artifact references)
- **Constraints**: Pane layout defaults vertical_split; refresh_hz capped to avoid exceeding render budget.

### Export Artifact
- **Purpose**: Captures combined Pygame view + chart viewport at a time window.
- **Fields**:
  - `artifact_id` (string)
  - `episode_id` (string)
  - `range` (object: `start_frame`, `end_frame`)
  - `type` (enum: screenshot|clip)
  - `path` (string under `output/`)
  - `created_at` (timestamp)
- **Constraints**: Exports reference telemetry stream and frame range; stored under artifact policy paths.

### Simulation Session (state machine)
- **States**: running → paused ↔ running → terminated/truncated.
- **Transitions**: pause/resume toggles; termination/truncation ends sampling; headless runs skip UI but keep state machine identical.

## Relationships
- Telemetry Stream contains ordered Telemetry Samples for one `episode_id`.
- Visualization Session consumes a Telemetry Stream (live tail or replay) and controls Export Artifacts.
- Export Artifacts are derived from a Visualization Session and reference the originating Telemetry Stream/episode.
