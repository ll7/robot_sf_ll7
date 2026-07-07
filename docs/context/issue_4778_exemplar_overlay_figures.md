# Issue #4778 — Exemplar Selection + Multi-Planner Overlay Integration

## Purpose

This note documents the integration between the three figure-generation
capabilities for campaign episode review:

1. **Exemplar episode auto-selection** (`robot_sf/benchmark/exemplar_selection.py`,
   `scripts/select_exemplar_episodes.py`): Picks representative episodes per
   (planner × mechanism/outcome) cell — median, best, worst by a chosen metric —
   and emits a deterministic selection manifest.

2. **Replay bridge** (`robot_sf/benchmark/episode_replay_figure.py`,
   `scripts/replay_episode_figure.py`): Takes a persisted campaign episode row
   and replays it to produce provenance-stamped stills, filmstrip, and trajectory
   figures.

3. **Multi-planner trajectory overlay** (`robot_sf/benchmark/multi_planner_overlay.py`,
   `scripts/render_multi_planner_trajectory_overlay.py`): Overlays robot
   trajectories from multiple planners on the same scenario+seed into a single
   provenance-stamped comparison figure.

## Data Flow

```
campaign/episodes.jsonl
        │
        ├──► select_exemplar_episodes.py ──► selection_manifest.json
        │                                        │
        ▼                                        ▼
 replay_episode_figure.py              render_multi_planner_trajectory_overlay.py
   (single episode replay)               --selection-manifest <manifest.json>
        │                                        │
        ▼                                        ▼
 stills / filmstrip / trajectory         multi-planner overlay figure
```

## Integration Points

### replay_steps → overlay extraction

Episode records carrying `replay_steps` data (from the replay bridge) are now
supported by `extract_trajectory_from_episode()`. The function first tries
inline trajectory fields (`trajectory.robot_positions`, etc.), then falls back
to extracting `(x, y)` from `replay_steps` entries.

### ReplayEpisode → TrajectoryRow bridge

`trajectory_row_from_replay_episode()` converts a `ReplayEpisode` object into a
`TrajectoryRow` for use with the overlay renderer. This enables programmatic
pipelines where replay data is loaded in-process rather than through JSONL.

### Selection manifest → overlay CLI

`scripts/render_multi_planner_trajectory_overlay.py` accepts
`--selection-manifest <path>` to drive overlay generation from an exemplar
selection manifest. The CLI resolves `(scenario_id, seed, planner_keys)` from
manifest entries and renders overlays for all cells in the manifest.

## Claim Boundary

All three tools produce **descriptive visualizations only** — they are not
benchmark evidence or metric claims. Provenance sidecars record the claim
boundary explicitly.

## Dependencies

- **#4777 (style pack)**: merged — provides shared planner palette and
  publication style.
- **#4776 (replay bridge)**: merged — provides `ReplayEpisode`/`ReplayStep`
  data structures and figure generation.

## Merged PRs

- PR #4788: Exemplar selection (Part 1)
- PR #4808: Multi-planner overlay (Part 2)
- PR #4796: Replay bridge (#4776)
- PR #4786: Style pack (#4777)
