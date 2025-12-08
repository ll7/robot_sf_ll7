# Feature Specification: Data Visualization alongside Pygame

**Feature Branch**: `343-telemetry-viz`  
**Created**: December 08, 2025  
**Status**: Draft  
**Input**: User description: "Data visualization alongside pygame" (issue #343)

## Clarifications

### Session 2025-12-08

- Q: How should the telemetry panel be integrated alongside Pygame? → A: Docked pane inside the same SDL/Pygame window, blitting pre-rendered chart images.

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Live Pygame + Telemetry panel (Priority: P1)

A researcher runs a simulation with the Pygame viewer and keeps a synchronized telemetry panel open beside it showing step FPS, episode reward, collision counters, and min pedestrian distance in real time.

**Why this priority**: This is the core promise of the feature: immediate situational awareness without leaving the Pygame window.

**Independent Test**: Start a single episode with the telemetry panel enabled; verify charts update at ≥1 Hz without dropping Pygame frame rate below the documented baseline.

**Acceptance Scenarios**:

1. **Given** a running episode with Pygame visible, **When** the user enables the telemetry panel, **Then** the charts render live metrics (FPS, reward, collisions, min distance) and stay in sync with the current timestep.
2. **Given** the telemetry panel is open, **When** the user pauses or resumes the simulation, **Then** the charts pause/resume accordingly without data loss or UI freezes.

---

### User Story 2 - Replay with synchronized charts (Priority: P2)

An analyst loads a saved episode (frames + telemetry stream) and scrubs through the timeline; the Pygame playback stays aligned with the charts, and exporting a still/clip also captures the corresponding metric slice.

**Why this priority**: Replay and exportability make the visualization useful for debugging and reporting beyond live observation.

**Independent Test**: Open a previously recorded run from `output/` artifacts and confirm playback controls sync charts and rendering without requiring a live simulator.

**Acceptance Scenarios**:

1. **Given** a completed run with recorded frames and telemetry, **When** the analyst opens it in replay mode, **Then** frame position, timestep, and chart cursor align within one frame of each other during scrubbing.
2. **Given** the analyst exports a short clip or screenshot, **When** they inspect the export, **Then** the chosen time range and chart annotations match the on-screen state at export time.

---

### User Story 3 - Headless/CI telemetry sanity (Priority: P3)

A QA engineer runs the simulator in headless mode (no Pygame window) and still receives a telemetry artifact plus a minimal static plot so CI can validate chart generation without a display server.

**Why this priority**: Ensures the visualization pipeline does not regress headless workflows and can be validated in automation.

**Independent Test**: Execute a headless smoke test with the telemetry panel disabled and confirm that telemetry JSONL plus an auto-generated PNG/JSON summary are written to the artifact root.

**Acceptance Scenarios**:

1. **Given** `SDL_VIDEODRIVER=dummy` and no display, **When** the headless smoke test runs with telemetry recording enabled, **Then** a telemetry file and summary plot are produced in `output/` and the run exits cleanly.
2. **Given** a CI run, **When** the artifact guard checks the outputs, **Then** it finds the telemetry summary and no missing/empty files even without rendering.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

- Extremely long episodes (tens of thousands of steps) must not exhaust memory in the telemetry buffer; decimation or chunked writes are required.
- Low frame-rate environments (≤5 FPS) must keep the charts responsive without over-sampling or freezing the UI.
- Missing metrics in legacy runs should degrade gracefully (e.g., show "not recorded" placeholders) without crashing the viewer.
- Multiple runs opened in succession should not leak file handles or render stale data in a new session.
- User closes the telemetry panel while Pygame keeps running; the simulator must continue normally and allow re-opening the panel without restart.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Provide a side-by-side or docked telemetry panel within the same SDL/Pygame window that renders live charts for FPS, reward, collision count, min pedestrian distance, and action norm with refresh ≥1 Hz by blitting pre-rendered chart images.
- **FR-002**: Expose a programmatic API (no CLI dependency) to enable/disable the telemetry panel and select metrics for live runs created via `make_robot_env` or related factories.
- **FR-003**: Support replay mode that reads recorded frames plus telemetry streams from `output/` artifacts and keeps chart cursors aligned to frame indices within one frame tolerance.
- **FR-004**: Persist telemetry to JSONL (or equivalent append-only format) during live runs and write synchronized metadata (timestamps, frame indices, episode ids) to allow deterministic replays.
- **FR-005**: Implement buffering/decimation so the telemetry pipeline adds <10% overhead to the baseline Pygame frame rate on the benchmark scenario used in CI/perf smoke tests.
- **FR-006**: Allow users to export screenshots or short clips that capture both the Pygame view and the current chart viewport, saving them under `output/` with stable naming.
- **FR-007**: Provide a headless fallback that, when Pygame is disabled, still emits telemetry JSONL and at least one summary graphic (PNG or SVG) without requiring a display server.
- **FR-008**: Surface health indicators in the UI/logs when telemetry drops samples (e.g., backpressure, write failures) and include them in the run manifest/summary for postmortem review.
- **FR-009**: Document the workflow in `docs/` and add/extend tests or smoke scripts covering live mode, replay mode, and headless mode artifact generation.

### Key Entities *(include if feature involves data)*

- **Telemetry Sample**: Per-timestep record containing timestamp, frame index, metrics (FPS, reward, collisions, min distance, action norm), and episode identifiers.
- **Telemetry Stream**: Append-only sequence of telemetry samples persisted to JSONL and optionally decimated for UI rendering.
- **Visualization Session**: UI/session state for either live or replay mode, including selected metrics, playback position, panel layout, and export targets.

### Implementation Constraints

- Telemetry charts must render off-screen (e.g., matplotlib/agg or equivalent) and be blitted into a docked pane inside the existing SDL/Pygame window to maintain synchronization and headless/CI parity without spawning extra windows or web frontends.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In live mode, enabling the telemetry panel reduces average Pygame FPS by no more than 10% compared to baseline in the CI performance smoke test.
- **SC-002**: Replay mode keeps frame and chart cursors aligned within one frame across a 5-minute recording, verified by automated assertions in the replay test harness.
- **SC-003**: Headless runs produce non-empty telemetry JSONL and a summary graphic in `output/` with zero missing-file violations reported by the artifact guard script.
- **SC-004**: At least one documented example/demo shows side-by-side Pygame and telemetry panel usage, and corresponding instructions are present in `docs/` and referenced from `docs/README.md`.
