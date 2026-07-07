# Issue #4776: Episode Replay Figure Bridge

## Purpose

This module provides a CPU-only bridge from persisted campaign episode rows to replay-derived figure artifacts: stills, filmstrip, and trajectory plots, all with deterministic replay checks and provenance sidecars.

**Claim boundary**: Figure artifact generation from retained episode rows only. This tool does **not** run campaigns, reinterpret metrics, or promote replay outputs as new benchmark evidence. Re-simulation is used solely to render a specific already-recorded episode row.

## Required Episode Fields

Episode rows must contain at minimum:

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | string | Unique episode identifier |
| `scenario_id` | string | Scenario identifier |
| `seed` | integer | Random seed used for the episode |

Optional provenance fields (recommended for full provenance tracking):

| Field | Type | Description |
|-------|------|-------------|
| `planner` | string | Planner name |
| `planner_key` | string | Alternative planner identifier |
| `algo` | string | Algorithm name |
| `algo_config` | object | Algorithm configuration |
| `campaign_id` | string | Campaign identifier |
| `campaign_root` | string | Campaign root path |
| `scenario_matrix_path` | string | Path to scenario matrix |
| `config_hash` | string | Campaign config hash |
| `scenario_matrix_hash` | string | Scenario matrix hash |
| `repo_commit` | string | Git commit hash |
| `replay_steps` | array | Replay trajectory data |
| `replay_dt` | float | Replay timestep |
| `replay_map_path` | string | Map SVG path |
| `final_robot_position` | array | Final robot [x, y] position |
| `final_progress` | float | Final progress value |
| `success` | boolean | Success flag |
| `collision` | boolean | Collision flag |

## Determinism Requirement

By default, the replay figure generation performs a determinism check comparing the replay output to the source episode row:

1. **Final robot position**: If `final_robot_position` is recorded in the episode row, the replay endpoint must match within the specified tolerance (default: 0.1m).
2. **Final progress**: If `final_progress` is recorded, it is checked for presence.
3. **Failure behavior**: If the determinism check fails, the tool raises a `RuntimeError` with details about the drift.

**Diagnostic escape hatch**: Use `--no-determinism-check` to skip determinism validation. This should only be used for diagnostic purposes and the resulting artifacts should **not** be treated as validated evidence.

## Example Command

Generate all figure types for a specific episode:

```bash
uv run python scripts/replay_episode_figure.py \
  --episodes output/campaign/episodes.jsonl \
  --episode-id ep_001 \
  --outputs still,filmstrip,trajectory \
  --out-dir output/replay_episode_figure/ep_001
```

Generate trajectory plot only with custom tolerance:

```bash
uv run python scripts/replay_episode_figure.py \
  --episodes episodes.jsonl \
  --episode-id ep_002 \
  --outputs trajectory \
  --out-dir output/trajectory_only \
  --tolerance-m 0.05
```

Generate filmstrip with custom frame steps in PDF format:

```bash
uv run python scripts/replay_episode_figure.py \
  --episodes episodes.jsonl \
  --episode-id ep_003 \
  --outputs filmstrip \
  --out-dir output/filmstrip \
  --frame-steps 0,10,20,30 \
  --format pdf
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--episodes <path>` | Path to episodes JSONL file (required) |
| `--episode-id <id>` | Episode ID to generate figures for (required) |
| `--outputs <types>` | Comma-separated list: still,filmstrip,trajectory (required) |
| `--out-dir <path>` | Output directory for artifacts (required) |
| `--campaign-root <path>` | Campaign root for resolving manifest metadata |
| `--scenario-matrix <path>` | Path to scenario matrix file |
| `--config <hash>` | Campaign config hash |
| `--tolerance-m <float>` | Determinism check tolerance in meters (default: 0.1) |
| `--frame-steps <steps>` | Comma-separated step indices for filmstrip |
| `--format <fmt>` | Output format: png, pdf, svg (default: png) |
| `--no-determinism-check` | Skip determinism check (diagnostic only) |
| `--verbose` | Enable verbose logging |

## Generated Artifacts

### Still Frame (`still_<step>.<fmt>`)

Single frame at a specific timestep showing robot and pedestrian positions.

### Filmstrip (`filmstrip.<fmt>`)

Multiple frames arranged horizontally showing episode progression.

### Trajectory Plot (`trajectory.<fmt>`)

Full trajectory showing robot path (blue line) and pedestrian paths (dashed lines), with start (green) and end (red) markers.

### Provenance Sidecar (`replay_provenance.json`)

Machine-readable JSON containing:

```json
{
  "campaign_id": "camp_123",
  "episode_id": "ep_001",
  "scenario_id": "crossing_easy",
  "seed": 42,
  "planner_key": "social_force",
  "scenario_matrix_path": "/path/to/matrix.json",
  "campaign_config_hash": "abc123...",
  "repo_commit": "deadbeef...",
  "replay_command": "uv run python scripts/replay_episode_figure.py ...",
  "determinism_check_status": "pass",
  "determinism_tolerance": 0.1,
  "source_episodes_jsonl_path": "/path/to/episodes.jsonl",
  "source_episodes_jsonl_sha256": "sha256hash...",
  "artifacts": [
    {"type": "trajectory", "path": "trajectory.png", "format": "png", "sha256": "..."}
  ],
  "generated_at": "2024-01-01T00:00:00+00:00"
}
```

### Caption Fragment (`caption_fragment.tex`)

LaTeX-ready caption text for inclusion in papers or reports:

```tex
Episode ep_001 (scenario: crossing_easy, seed: 42, planner: social_force). Determinism check: pass. Position error: 0.023m.
```

## Provenance Semantics

Every artifact includes:

1. **Machine-readable provenance**: JSON sidecar with full metadata
2. **Visual stamp**: Corner stamp or caption text on figures with campaign ID, episode ID, seed, and planner
3. **SHA-256 hashes**: For both source episodes file and generated artifacts
4. **Replay command**: Exact command line used for reproduction
5. **Determinism status**: Pass/fail/skipped with tolerance and error details

## Claim Boundary

**IMPORTANT**: Artifacts generated by this tool are **figure artifacts only**. They are:

- Suitable for visualization, debugging, and paper figures
- Derived from already-recorded episode rows
- Not new benchmark evidence
- Not a substitute for running full campaigns
- Not a reinterpretation of metrics

Do not use replay-derived figures as evidence of benchmark performance unless the source episode row itself is valid benchmark evidence with full provenance.

## Implementation Details

### Replay Episode Construction

The tool reads `replay_steps` from the episode row, which should contain per-timestep state:

```json
{
  "replay_steps": [
    {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "speed": 0.5, "ped_positions": [[1.0, 2.0]]},
    {"t": 1.0, "x": 1.0, "y": 0.5, "heading": 0.1, "speed": 0.5, "ped_positions": [[1.2, 2.1]]}
  ]
}
```

Steps can also be in tuple format: `[t, x, y, heading]`.

### Determinism Check Algorithm

1. Extract final robot position from replay (last step x, y)
2. Compare to `final_robot_position` from episode row
3. Compute Euclidean distance
4. Pass if distance <= tolerance, fail otherwise

### Figure Generation

- Uses matplotlib with Agg backend (headless-safe)
- All figures are CPU-only, no GPU required
- Map background is optional (loaded from `replay_map_path` if available)
- Pedestrian trajectories shown when `ped_positions` available in replay steps

## Testing

Run tests with:

```bash
uv run pytest tests/benchmark/test_episode_replay_figure.py -v
```

Key test coverage:

- Episode row validation (required/optional fields)
- Replay episode construction from various formats
- Determinism check pass/fail/not_evaluable cases
- Figure generation for all output types
- Provenance sidecar content validation
- CLI argument parsing and error handling
- Failure modes for corrupted/missing data

## Related Files

- `robot_sf/benchmark/episode_replay_figure.py`: Core module
- `scripts/replay_episode_figure.py`: CLI entry point
- `tests/benchmark/test_episode_replay_figure.py`: Test suite
- `robot_sf/benchmark/full_classic/replay.py`: Replay data structures
- `robot_sf/benchmark/full_classic/visuals.py`: Visualization pipeline
