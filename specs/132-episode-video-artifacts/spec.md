# Feature Specification: Episode Video Artifacts MVP

**Feature Branch**: `132-episode-video-artifacts`  
**Created**: 2025-09-24  
**Status**: Draft  
**Input**: User description: "Episode Video Artifacts MVP: add per-episode MP4 generation (synthetic or SimulationView renderer), --no-video toggle, JSON Schema manifest entries, micro-batch integration test, and performance budget (<5% overhead); document design and link in docs."

## User Scenarios & Testing

### Primary User Story
As a benchmark user running evaluation batches, I want each episode to optionally produce a lightweight MP4 so I can quickly review qualitative behavior, while keeping runs reproducible, fast, and easy to disable when I only need metrics.

### Acceptance Scenarios
1. Given a micro batch (1–2 episodes) with video enabled, When the batch completes, Then MP4 files exist for those episodes, each with file size > 0 and frame count equal to the number of recorded steps.
2. Given a run with the --no-video toggle enabled (or equivalent config), When the batch completes, Then no MP4 files are produced and no video-related work is performed.
3. Given a run on macOS and headless CI environments, When video generation is enabled, Then video artifacts are produced reliably without backend-dependent failures.
4. Given a large batch run, When video generation is enabled, Then the overall overhead attributable to encoding is below the documented performance budget.
5. Given existing benchmark outputs, When aggregating or validating episodes, Then video artifact metadata is represented in the manifest/schema and validates successfully.

### Edge Cases
- What happens when MoviePy or a system codec is unavailable? Videos should be skipped with a clear message and the run should still succeed.
- How does the system behave if no frames are captured (empty episode)? It should produce no MP4 and record a reason, not crash.
- How is the renderer chosen when SimulationView is not available? The synthetic renderer should be used automatically.
- What if output paths aren’t writable? Fail the episode’s video step gracefully with a clear error and continue the batch.

## Requirements

### Functional Requirements
- FR-001: System MUST support per‑episode MP4 generation as an optional feature.
- FR-002: System MUST provide a user‑facing toggle to disable video generation (e.g., `--no-video` or config flag) across CLI and programmatic APIs.
- FR-003: System MUST support at least two frame sources: a synthetic renderer and, when available, a SimulationView‑based renderer; selection MUST be deterministic and documented.
- FR-004: System MUST emit video artifact metadata in a JSON/JSONL manifest/schema, including path, format (mp4), and file size (>0 validation rule).
- FR-005: System MUST include a micro‑batch integration test that asserts MP4 presence, non‑zero size, and frames == steps for at least one episode.
- FR-006: System MUST log clear skip reasons when video cannot be generated (missing MoviePy, missing frames, unwritable path) without failing the overall batch.
- FR-007: System MUST keep video overhead within the stated performance budget under default settings.
- FR-008: System MUST document the design, usage, performance expectations, and link it from the docs index and benchmark TODO.
- FR-009: System MUST be reproducible (deterministic file naming strategy tied to episode identity) and compatible with resume behavior.
- FR-010: System MUST allow per‑run selection of renderer mode via a stable flag (e.g., `--video-renderer=synthetic|sim-view|none`).

- FR-011: System MUST handle absent frames by not creating empty video files and by writing a structured log/warning with context (episode id, renderer, step count).
- FR-012: System MUST ensure that enabling video does not change core metrics or episode control flow (non‑intrusive recording path).

- FR-013: System MUST update the episode JSON Schema to include video metadata and ensure CI validates these fields when present (format='mp4', filesize_bytes>0, frames≥0).
- FR-014: System MUST record and publish a small performance sample in the documentation (encode ms/frame and % overhead) including hardware/OS/context, and link it from the docs index and benchmark TODO.
- FR-015: System MUST run the micro‑batch video test in CI as a smoke guard (skip on platforms lacking codecs if needed) to prevent regressions.

## Clarifications

### Session 2025-09-24 — Options accepted
- CLI flags and naming
	- Video toggle: `--no-video` (boolean flag)
	- Renderer selector: `--video-renderer=synthetic|sim-view|none`
	- Programmatic API option names: `video_enabled: bool`, `video_renderer: Literal['synthetic','sim-view','none']`
- Defaults and behavior
	- Default video generation: Disabled by default; users enable explicitly
	- Default renderer: `synthetic` (use `sim-view` only if explicitly selected)
- Output location and naming
	- Output directory: `results/videos/` under the run’s output stem
	- File naming: `<scenario>__<seed>__<episode_idx>.mp4`
	- Determinism: Overwrite by default with resume-aware checks
- Manifest/schema fields
	- Placement: In-episode JSONL record
	- Fields: `video: { path, format: 'mp4', filesize_bytes, frames, renderer, notes? }`
- Performance budget policy
	- Target: <5% overhead; WARN by default, FAIL only when `ROBOT_SF_PERF_ENFORCE=1` is set
- Headless/codec resilience
	- If MoviePy/codec missing: Skip video with structured warning; batch succeeds
- Test scope
	- Micro-batch test: 1–2 episodes asserting existence, size>0, frames==steps

### Key Entities
- Episode Video Artifact: Represents the per‑episode MP4 output with attributes: episode_id, file_path, format, filesize_bytes, renderer, frames, notes/skips.
- Renderer Mode: A small enumerated domain controlling the frame source; defaults and fallbacks documented.
- Performance Budget: Thresholds for acceptable overhead; how measured and where reported/documented.

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed
