# Requirements Checklist: Data Visualization alongside Pygame

**Purpose**: Track readiness of live/replay telemetry visualization adjacent to the Pygame viewer.
**Created**: December 08, 2025
**Feature**: [spec.md](../spec.md)

## Live Telemetry Panel

- [ ] CHK001 Telemetry panel renders FPS, reward, collision count, min pedestrian distance, and action norm at ≥1 Hz while Pygame runs.
- [ ] CHK002 Panel can be enabled/disabled programmatically from environment factory entry points (no CLI-only toggle).
- [ ] CHK003 Pause/resume of the simulation keeps charts synchronized without data gaps or UI hangs.

## Replay Mode

- [ ] CHK010 Replay opens recorded frames + telemetry from `output/` artifacts and keeps chart cursor aligned within one frame during scrubbing.
- [ ] CHK011 Exporting a screenshot or short clip captures both the Pygame view and the chart viewport for the selected time span.

## Headless/CI

- [ ] CHK020 Headless runs (e.g., `SDL_VIDEODRIVER=dummy`) still emit telemetry JSONL plus a summary graphic in `output/`.
- [ ] CHK021 Artifact guard/validation reports zero missing or empty telemetry outputs in CI.

## Performance & Resilience

- [ ] CHK030 Telemetry pipeline adds <10% FPS overhead relative to baseline performance smoke tests.
- [ ] CHK031 Telemetry backpressure or dropped samples surface as warnings/indicators and are recorded in the run manifest/summary.

## Documentation & Tests

- [ ] CHK040 A documented example/demo shows side-by-side Pygame + telemetry panel usage and is linked from `docs/README.md`.
- [ ] CHK041 Tests or smoke scripts cover live mode, replay mode, and headless artifact generation.
