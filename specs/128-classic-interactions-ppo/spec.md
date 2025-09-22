# Feature Specification: Classic Interactions PPO Pygame Visualization

**Feature Branch**: `128-classic-interactions-ppo`  
**Created**: 2025-09-22  
**Status**: Draft  
**Input**: User description: "Classic interactions PPO pygame visualization: run scenarios from configs/scenarios/classic_interactions.yaml using SimulationView (sim_view.py) and default PPO model (model/ppo_model_retrained_10m_2025-02-01.zip); provide example script and benchmark loader utility; Display the simulation of the benchmark and enable video recording by default with a toggle."

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a researcher evaluating robot navigation behavior in canonical pedestrian interaction scenarios, I want to launch an interactive (or headless) visualization that replays benchmark scenarios from the classic interactions matrix while a pre‑trained PPO policy controls the robot, so that I can qualitatively inspect decision making, capture illustrative videos, and rapidly iterate on scenario or policy adjustments.

### Acceptance Scenarios
1. **Given** the repository with dependencies installed and model file present, **When** the user runs the visualization entry point with default parameters, **Then** the first scenario in `classic_interactions.yaml` loads, the PPO policy controls the robot, and a window (or headless fallback) displays live simulation frames until episode termination.
2. **Given** the user provides a specific scenario name, **When** they start the visualization, **Then** only that scenario’s episodes (its defined seeds) are cycled through sequentially with clear labeling (scenario name, episode seed, outcome) on the display.
3. **Given** the user enables a video recording toggle, **When** an episode finishes, **Then** an MP4 file is produced (or a clear skip note is reported if recording dependencies are unavailable) without interrupting the next episode playback.
4. **Given** the model file is missing, **When** the user starts the visualization, **Then** a clear error message explains the missing dependency and no simulation loop begins.
5. **Given** the user runs in an environment without display (headless), **When** they start the visualization, **Then** the system automatically falls back to off‑screen rendering (or recording only) and logs this mode selection.
6. **Given** the user interrupts (Ctrl+C / window close), **When** the current episode ends prematurely, **Then** resources are released cleanly and any completed episodes retain their recordings without corruption.
7. **Given** the user specifies a maximum number of episodes, **When** the limit is reached, **Then** the program exits with a success status and summary.
8. **Given** a scenario with multiple seeds, **When** playback advances, **Then** seeds are processed in deterministic order as listed in the scenario file.

### Edge Cases
- Scenario name provided does not exist → feature must report available scenario names and exit gracefully.
- Model file path incorrect → clear actionable error (shows expected path) before any environment creation.
- Video recording enabled but ffmpeg/moviepy unavailable → episodes still play; each episode logs a skipped recording with reason.
- Headless environment detection ambiguous (e.g., partial pygame init failure) → fallback path warns and proceeds without interactive controls.
- Very long scenario list (user overrides matrix) → feature respects explicit max episodes limit to avoid unintended long runs.
- Episode terminates via collision/timeout → overlay/outcome labeling still shown for final frame before advancing.
- User sets recording directory unwritable → recording is skipped with explicit permission error note; simulation continues.

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: Feature MUST load scenario definitions from the classic interactions matrix file at a user‑selectable path (default `configs/scenarios/classic_interactions.yaml`).
- **FR-002**: Feature MUST allow selecting a single scenario by name or default to the first listed when unspecified.
- **FR-003**: Feature MUST iterate deterministically through each seed declared for the chosen scenario(s) in listed order.
- **FR-004**: Feature MUST load a pre‑trained PPO policy from a user‑configurable filepath (default `model/ppo_model_retrained_10m_2025-02-01.zip`).
- **FR-005**: Feature MUST execute an episode loop where robot actions are produced via the loaded policy until termination (success, collision, timeout, or other terminal condition signaled by the environment).
- **FR-006**: Feature MUST render each timestep using the simulation visualization layer with scenario and episode metadata overlay (scenario name, seed, timestep, optional outcome when terminal).
- **FR-007**: Feature MUST provide an option to enable or disable MP4 recording per episode via a flag or configuration parameter.
- **FR-008**: When recording is enabled and dependencies are present, feature MUST write one MP4 file per episode using a deterministic naming convention including scenario name and seed.
- **FR-009**: When recording prerequisites are missing (e.g., moviepy/ffmpeg), feature MUST skip recording and log a human‑readable reason without aborting playback.
- **FR-010**: Feature MUST allow limiting total episodes executed via a numeric parameter (e.g., `--max-episodes`).
- **FR-011**: Feature MUST gracefully handle user interruption (Ctrl+C or window close) by releasing resources and closing any open visualization surface.
- **FR-012**: Feature MUST detect headless contexts and automatically disable interactive window creation while still permitting optional recording or frame capture.
- **FR-013**: Feature MUST summarize run results on completion (episodes attempted, completed, recorded, skipped recordings reasons aggregated).
- **FR-014**: Feature MUST fail fast with a clear message if the PPO model file is not found at provided path.
- **FR-015**: Feature MUST expose a simple public entry point callable programmatically (not only via CLI) for integration into notebooks or scripts.
- **FR-016**: Feature MUST surface outcome category for each episode (success, collision, timeout, other) for later qualitative analysis.
- **FR-017**: Feature SHOULD allow optional output directory override for recordings and logs.
- **FR-018**: Feature SHOULD provide a flag to disable overlays (minimal visual mode) for performance-sensitive usage.
- **FR-019**: Feature SHOULD continue processing subsequent episodes even if one episode recording fails.
- **FR-020**: Feature SHOULD emit structured log lines (or structured summary object) enabling downstream tooling to compute simple statistics.
- **FR-021**: Feature SHOULD provide a dry‑run mode that validates scenario availability and model path without launching visualization.
- **FR-022**: Feature MAY support batching multiple scenarios sequentially if user supplies a list (clarify need) [NEEDS CLARIFICATION: multi-scenario sequential mode required or out-of-scope?].
- **FR-023**: Feature MAY provide a frame sampling rate option for lightweight previews (clarify granularity) [NEEDS CLARIFICATION: is frame decimation required?].

Unclear / pending confirmation requirements are explicitly marked.

### Key Entities
- **Scenario Definition**: Conceptual record containing a name, map reference, simulation parameters, metadata tags (archetype, density, flow, groups), and a list of seeds.
- **Episode**: A single seeded run of a scenario producing a trajectory, outcome category, optional recording artifact, and summary metrics (steps, termination cause).
- **Policy Model**: Pre‑trained navigation decision component loaded once and queried each timestep for an action given observation.
- **Recording Artifact**: Optional video file representing visual playback of a single episode, associated with scenario name and seed.
- **Run Summary**: Aggregated counts (episodes completed, recordings generated/skipped, reasons) and outcomes distribution.

## Review & Acceptance Checklist

### Content Quality
- [ ] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and mostly unambiguous  
- [x] Success criteria are measurable (episode counts, recording presence, error messages)
- [x] Scope is clearly bounded (single-scenario or limited sequential playback of matrix entries)
- [x] Dependencies and assumptions identified (model file, scenario yaml, visualization environment, optional ffmpeg)

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed (pending clarification answers)
