# Feature Specification: Integrate Plots & Videos into Full Classic Benchmark

**Feature Branch**: `126-title-integrate-plots`  
**Created**: 2025-09-20  
**Status**: Draft  
**Input**: User description: "Wire existing placeholder plot and video generation modules (plots.py, videos.py) into run_full_benchmark orchestrator. Add videos directory creation, invoke generation once after adaptive loop completion, write artifact manifests (plot_artifacts.json, video_artifacts.json) capturing status. Respect cfg.disable_videos, cfg.max_videos, smoke mode, and optional dependencies (matplotlib, moviepy). Update docs (dev issue folder) and changelog entry. Provide tests: ensure artifacts created or skipped markers. Non-goals: real trajectory rendering, rich metrics overlays. Risks: optional deps missing, performance overhead. Config: reuse existing flags no new ones unless necessary."

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a benchmark consumer (researcher or CI reviewer), after running the Full Classic Interaction Benchmark, I want automatically generated visual artifacts (summary plots and representative episode videos or explicit skipped markers) so that I can quickly assess qualitative behavior, detect regressions, and attach visuals to reports without manual post‑processing.

### Acceptance Scenarios
1. **Given** I run the benchmark with default settings (non-smoke, videos enabled), **When** it completes, **Then** the output root contains `plots/*.pdf`, a `videos/` folder with up to the configured number of MP4 files, and JSON manifest files enumerating each artifact with status `generated` (or `skipped` if deps missing) without errors.
2. **Given** I run the benchmark with `--disable-videos`, **When** it completes, **Then** the video artifact manifest lists selected episodes with status `skipped` and note indicating videos disabled, and no MP4 files are produced.
3. **Given** I run the benchmark in smoke mode, **When** it completes, **Then** placeholder plot PDFs still exist (or are marked skipped only if plotting dependency absent) and videos are all skipped with a smoke note.
4. **Given** optional dependencies (matplotlib or moviepy/ffmpeg) are not installed, **When** the benchmark runs, **Then** the manifests show `skipped` with a reason ("matplotlib missing" or "moviepy missing") and the run otherwise succeeds.
5. **Given** I re-run the benchmark in the same output directory (resume), **When** visuals already exist, **Then** plots and videos are regenerated only once per completed run stage (no duplication) and existing episode JSON outputs remain unchanged.

### Edge Cases
- Missing optional dependency: artifacts appear with `skipped` status and explanatory note; run does not fail.
- Zero episodes (e.g., early termination): manifests exist but may have empty artifact lists; still considered success.
- High `max_videos` vs small episode count: only available episodes (bounded by count) are selected; no error.
- Videos disabled plus missing dependencies: note prioritizes disabled status (implementation detail: requirement is final manifest clarity; user just sees disabled reason).
- Concurrent multi-worker run: visual artifact generation remains deterministic and based on final aggregated records, not partial intermediate state.

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: System MUST generate a deterministic set of plot artifact metadata after the benchmark run completes (or mark each plot as skipped with a clear reason).
- **FR-002**: System MUST generate up to a configured maximum number of representative episode video artifacts (status generated or skipped) based on episodes collected during the run.
- **FR-003**: System MUST respect a user flag to disable video creation, producing skip entries instead of errors.
- **FR-004**: System MUST operate correctly in smoke mode by skipping video generation while still producing (or attempting) lightweight plot placeholders.
- **FR-005**: System MUST expose artifact metadata in machine-readable JSON files (one for plots, one for videos) for downstream automation.
- **FR-006**: System MUST ensure benchmark completion is not blocked by missing optional visualization dependencies; such cases MUST be recorded as skipped artifacts.
- **FR-007**: System MUST limit video selection deterministically (first N eligible episodes) to guarantee reproducibility.
- **FR-008**: System MUST not introduce breaking changes to existing JSON summary, effect size, or precision artifacts.
- **FR-009**: System MUST place all visual outputs under the benchmark output root, segregated into `plots/` and `videos/` directories.
- **FR-010**: System MUST include clear human-readable notes in artifact metadata explaining skipped or error statuses.
- **FR-011**: System MUST allow subsequent analytical tooling to proceed even if all artifacts are skipped.
- **FR-012**: System MUST avoid unbounded growth by generating visuals only once per completed benchmark run (not per adaptive iteration) to minimize overhead.
- **FR-013**: System MUST record video artifacts only for existing episodes at time of finalization (no speculative or empty videos).
- **FR-014**: System SHOULD keep added runtime overhead minimal (target: plots < 2s total, videos < 5s for 1 default video) for typical CI usage. [NEEDS CLARIFICATION: precise performance thresholds acceptance?]
- **FR-015**: System SHOULD maintain deterministic artifact filenames for reproducibility.

### Key Entities
- **Plot Artifact**: Logical descriptor of a generated or skipped plot (kind, file path, status, note). Used for downstream reporting.
- **Video Artifact**: Descriptor of a representative episode video (episode id, scenario id, file path, status, note). Enables lightweight visual regression checks.
- **Benchmark Output Root**: Directory chosen by user where all benchmark products (episodes, aggregates, reports, plots, videos) reside.
- **Benchmark Configuration Flags**: Existing parameters controlling mode (smoke), disabling videos, and maximum count for representative videos.

### Existing Visualization Context (Informational)
The project already includes an interactive / programmatic visualization layer using PyGame (`SimulationView` in `robot_sf/render/sim_view.py`). That system:
- Renders real simulation frames with robot, pedestrians, sensors, actions.
- Optionally records frame sequences to video (if moviepy/ffmpeg available) when explicitly enabled during simulation runs.

This feature (plots & representative videos for the classic benchmark) is intentionally distinct:
- Focuses on lightweight, post-run summary artifacts for reporting, not interactive playback.
- Uses synthetic or placeholder representations (deterministic paths, summary PDFs) to keep CI overhead low.
- Does not depend on opening a display window or running live simulation loops.

Clarification: We are NOT replacing or extending the PyGame `SimulationView` here; we are adding benchmark-scoped artifact generation triggered after statistical aggregation. Future work could integrate higher-fidelity frame capture from `SimulationView`, but that is explicitly out of scope (see Non‑Goals).

## Review & Acceptance Checklist

### Content Quality
- [ ] No implementation details (languages, frameworks, APIs) beyond what is necessary to define user-visible behavior
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders (behavioral framing)
- [x] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain (one present re: perf threshold)
- [x] Requirements are testable and unambiguous (except noted clarification)
- [x] Success criteria are measurable (artifact presence & statuses)
- [x] Scope is clearly bounded (no high-fidelity rendering)
- [x] Dependencies and assumptions identified (optional plotting & video libs)

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (performance threshold)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed (pending clarification removal)

