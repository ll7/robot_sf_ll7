# Feature Specification: Enhance Benchmark Visual Artifacts (SimulationView Replay, Encoding, Schema Validation)

**Feature Branch**: `127-enhance-benchmark-visual`  
**Created**: 2025-09-20  
**Status**: Draft  
**Input**: User description: "Enhance benchmark visual artifacts: implement real SimulationView replay rendering and MP4 encoding (with moviepy/ffmpeg if available, graceful skip otherwise); add replay adapter extracting position, heading, timestep for episodes; introduce JSON Schema validation for plot_artifacts.json and video_artifacts.json plus performance_visuals.json; update documentation (benchmark visuals section) and demo script to reflect operational SimulationView path; add tests for successful SimulationView renderer and moviepy absence skip; maintain deterministic selection and performance budgets (<2s plots, <5s single video)."

## User Scenarios & Testing

### Primary User Story
A researcher runs the full classic benchmark and expects high‑fidelity representative episode videos rendered via the actual SimulationView (not synthetic fallback) along with validated, structured manifests. If optional dependencies (moviepy/ffmpeg or pygame) are missing, the system transparently records skipped artifact entries with clear reasons while preserving deterministic plot outputs.

### Acceptance Scenarios
1. Given a system with pygame + moviepy installed and episodes produced, When the benchmark completes, Then video artifacts are generated using SimulationView with renderer field = "simulation_view" and MP4 files exist.
2. Given moviepy is missing but pygame present, When the benchmark completes, Then video artifacts are all marked skipped with note "moviepy-missing" and no MP4 files are produced (no crash).
3. Given pygame (SimulationView) is unavailable, When the benchmark completes, Then synthetic fallback videos are produced (renderer = "synthetic") exactly as before.
4. Given JSON artifact manifests are written, When schema validation runs in dev/test mode, Then invalid structure would raise/log a validation error (and current valid manifests pass).
5. Given performance budgets (<2s plots, <5s first video) When a normal-size run (≤5 videos, default N) finishes, Then recorded performance_visuals.json contains flags indicating no soft budget breach.
6. Given deterministic episode ordering, When benchmark is run twice with identical seed & matrix, Then the chosen episode IDs for video rendering match exactly.

### Edge Cases
- Episode lacks required replay state fields → All corresponding video entries skipped with note "insufficient-replay-state".
- Partial failure while encoding frame k → Video artifact marked failed with note and partial file (if any) removed.
- Excessive episodes requested (max_videos > available episodes) → Only available subset used (no error).
- Performance over budget → Soft breach flagged; run still succeeds.
- Very large frame list (memory concern) → Frame iteration yields incremental encoding (streaming) rather than storing all frames (implementation detail intentionally not described here; requirement notes streaming need below).

## Requirements

### Functional Requirements
- **FR-001**: System MUST generate video artifacts using SimulationView when pygame is available and replay state is sufficient.
- **FR-002**: System MUST fall back to synthetic video generation when SimulationView import or initialization fails.
- **FR-003**: System MUST deterministically select the first N eligible episodes based on existing ordering and max_videos setting.
- **FR-004**: System MUST record each video artifact in `video_artifacts.json` with fields: episode_id, filename (if generated), renderer, status (success|skipped|failed), note (optional).
- **FR-005**: System MUST record each plot artifact in `plot_artifacts.json` with fields: name, filename (if generated), status, note (optional).
- **FR-006**: System MUST provide JSON Schema validation for plot, video, and performance manifests in development/test mode.
- **FR-007**: System MUST skip video generation gracefully when moviepy/ffmpeg is missing, marking artifacts with a clear note.
- **FR-008**: System MUST skip video generation with note "insufficient-replay-state" when required trajectory fields are absent.
- **FR-009**: System MUST enforce performance measurement and include timing + soft budget boolean flags in `performance_visuals.json`.
- **FR-010**: System MUST keep plot generation total wall time < 2 seconds for default matrix scale (soft budget; breach only flagged, not failing run).
- **FR-011**: System MUST keep a single SimulationView video wall time < 5 seconds under default conditions (soft budget; breach flagged).
- **FR-012**: System MUST stream frame encoding (not accumulate all frames in memory) for SimulationView videos when possible. [NEEDS CLARIFICATION: acceptable peak memory target?]
- **FR-013**: System MUST expose distinct skip notes for each degradation case (smoke-mode, disabled, simulation-view-missing, moviepy-missing, insufficient-replay-state).
- **FR-014**: System MUST ensure reproducible filenames (`video_<episode_id>.mp4`).
- **FR-015**: System MUST include at least one test asserting renderer == "simulation_view" in success path.
- **FR-016**: System MUST include a test asserting renderer == "synthetic" when SimulationView unavailable.
- **FR-017**: System MUST include a test asserting skip note == "moviepy-missing" when moviepy absent.
- **FR-018**: System MUST document the benchmark visuals workflow in repository docs (new section or page) including dependency matrix + degradation table.
- **FR-019**: System MUST update the demo script docstring to reflect SimulationView operational path.
- **FR-020**: System MUST leave existing episode/aggregate schemas unchanged.

### Key Entities
- **ReplayState**: Conceptual data needed for SimulationView reproduction (positions, headings, timestamps per timestep). Fields referenced; implementation fields not mandated here.
- **VideoArtifact**: Episode-level video outcome (episode_id, renderer, status, note, filename).
- **PlotArtifact**: Plot generation outcome (name, status, note, filename).
- **PerformanceVisuals**: Timing and budget evaluation (plots_time_s, first_video_time_s, plots_over_budget, video_over_budget).

## Review & Acceptance Checklist

### Content Quality
- [ ] No implementation details (Implementation hints present? Review FR-012 for potential detail leak.)
- [x] Focused on user value and business needs (research reproducibility & clarity)
- [x] Written for non-technical stakeholders (largely; some technical phrasing minimal)
- [x] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain (FR-012 needs resolution)
- [x] Requirements are testable and unambiguous (except performance nuance marked)
- [x] Success criteria are measurable
- [x] Scope is clearly bounded (visual artifact enhancement only)
- [x] Dependencies and assumptions identified (pygame, moviepy/ffmpeg, deterministic ordering)

## Execution Status
- [x] User description parsed
- [x] Key concepts extracted (SimulationView, replay adapter, encoding, schemas, performance budgets)
- [x] Ambiguities marked (FR-012)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed (pending FR-012 clarification)

