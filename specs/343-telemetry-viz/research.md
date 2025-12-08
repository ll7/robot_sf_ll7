# Research Notes: Data Visualization alongside Pygame

## Decisions

### Decision: Docked pane inside single SDL/Pygame window
- **Rationale**: Keeps synchronization simple, avoids multi-window focus issues, and preserves headless/CI parity without browser/toolkit dependencies.
- **Alternatives considered**: (1) Local web UI via websocket + browser (adds stack and focus juggling), (2) Separate matplotlib/Qt window (extra window management, weaker sync), (3) CLI-only summaries (no live UX).

### Decision: Telemetry persistence as append-only JSONL with frame indices
- **Rationale**: Matches existing artifact policy and resume semantics; easy to stream/decimate; deterministic alignment via `frame_idx` and timestamps.
- **Alternatives considered**: (1) Binary/npz streams (less debuggable), (2) SQLite/duckdb (heavier dependency), (3) CSV (lacks nested metadata, harder schema evolution).

### Decision: Off-screen chart rendering with matplotlib/agg and blitting
- **Rationale**: Pure-Python, deterministic rendering; integrates with existing matplotlib styles; off-screen keeps headless-compatible; blitting minimizes overhead.
- **Alternatives considered**: (1) pygame.draw primitives (limited charting), (2) vispy/GL (heavier GPU dependency), (3) browser-based plots (separate stack, less deterministic in CI).

### Decision: Headless summary artifact as PNG + JSON sidecar
- **Rationale**: PNG is easy to view in CI artifacts; JSON sidecar preserves numeric summary; both fit artifact policy under `output/`.
- **Alternatives considered**: (1) Only JSON (no quick visual), (2) Only SVG (less convenient for CI previews), (3) GIF/video (heavier, larger artifacts).
