# Implementation Plan: Data Visualization alongside Pygame

**Branch**: `343-telemetry-viz` | **Date**: December 08, 2025 | **Spec**: `specs/343-telemetry-viz/spec.md`
**Input**: Feature specification from `/specs/343-telemetry-viz/spec.md`

**Note**: Filled by `/speckit.plan` following `.github/prompts/speckit.plan.prompt.md`.

## Summary

Deliver a single-window Pygame experience with a docked telemetry pane that blits off-screen rendered charts (FPS, reward, collisions, min distance, action norm) in live runs, supports replay with cursor sync, and emits headless artifacts (JSONL + summary graphic) while keeping frame-rate overhead under 10%.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: Pygame/SDL, matplotlib (agg), NumPy, robot_sf.telemetry (manifest writer, sampler), Loguru  
**Storage**: Append-only JSONL telemetry + PNG/SVG summary artifacts under `output/` (respecting artifact policy)  
**Testing**: pytest (unified suite), headless smoke scripts; reuse CI perf smoke thresholds  
**Target Platform**: Linux/macOS desktops and CI headless (SDL_VIDEODRIVER=dummy)  
**Project Type**: Single Python library/application (simulation + visualization)  
**Performance Goals**: Pygame FPS degradation ≤10% vs baseline; replay cursor alignment within 1 frame over 5-minute runs  
**Constraints**: Single SDL window; no additional GUI toolkits or web frontends; off-screen chart rendering blitted into pane  
**Scale/Scope**: Episodes up to tens of thousands of steps; telemetry decimation prevents unbounded memory growth

## Constitution Check

GATE: PASS. Aligned with Principles II (factory use), VII (no contract breaks; reuse existing telemetry schema), VIII (docs entry + index link), IX (smoke + assertion tests for public behavior), XII (Loguru logging), XIII (test value verification for new tests). No scope creep beyond social-navigation visualization.

## Project Structure

### Documentation (this feature)

```text
specs/343-telemetry-viz/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/
```

### Source Code (repository root)

```text
robot_sf/telemetry/        # sampling, persistence, manifest helpers
robot_sf/render/           # Pygame rendering integration for docked pane
robot_sf/gym_env/          # factory toggles and wiring
tests/telemetry/           # live/replay/headless tests and smoke checks
examples/advanced/         # demo script showing side-by-side panel
docs/                      # new/updated guide linked from docs/README.md
output/                    # artifacts (JSONL, PNG/SVG summaries, exports)
```

**Structure Decision**: Single-codebase Python layout; integrate telemetry UI within existing Pygame/render modules, extend telemetry persistence, add focused tests/docs/examples under existing repo conventions.

## Complexity Tracking

No constitution violations requiring justification; keep single-window constraint and reuse existing modules.

## Phase 0: Outline & Research

- Extracted unknowns: none blocking; validate blitting approach, telemetry format, headless summary artifact.  
- Tasks: document decisions in `research.md`; confirm off-screen rendering pipeline and JSONL schema sufficiency; note alternatives (web UI, separate window).  
- Outputs: `research.md` (decisions, rationale, alternatives).

## Phase 1: Design & Contracts

- Derive entities/fields into `data-model.md` (Telemetry Sample/Stream, Visualization Session, Export Artifact).  
- Generate API/usage contract in `contracts/` (OpenAPI-style for programmatic toggles: enable live panel, replay, export).  
- Author `quickstart.md` with code snippets for live, headless, replay flows plus artifact locations.  
- Run `.specify/scripts/bash/update-agent-context.sh copilot` to record new tech/context.  
- Re-evaluate Constitution Check post-design (expected PASS; no contract breaks, docs/test coverage planned).

## Phase 2: Implementation Planning (post-/speckit.tasks)

- Defer granular task breakdown to `/speckit.tasks` after this plan; prioritize live pane, replay sync, headless outputs, exports, and tests/docs.
