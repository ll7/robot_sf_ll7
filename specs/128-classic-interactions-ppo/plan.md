# Implementation Plan: Classic Interactions PPO Pygame Visualization

**Branch**: `128-classic-interactions-ppo` | **Date**: 2025-09-22 | **Spec**: `/specs/128-classic-interactions-ppo/spec.md`
**Input**: Feature specification from `/specs/128-classic-interactions-ppo/spec.md`

## Summary
Provide an out-of-the-box, code-configured (no CLI) demonstration that replays one or more classic interaction scenarios using a pre-trained PPO policy with optional recording. Focus is qualitative inspection, deterministic seed ordering, and minimal external dependencies. Future extensibility: multi-scenario chaining, frame decimation, richer overlays.

## Technical Context
**Language/Version**: Python 3.13 (repo standard)  
**Primary Dependencies**: Stable-Baselines3 (PPO), Gymnasium, project env factories, SimulationView (pygame), moviepy (optional)  
**Storage**: Local filesystem (model zip, scenario YAML, optional MP4 outputs)  
**Testing**: pytest (add smoke tests)  
**Target Platform**: macOS/Linux dev machines; headless CI (recording skip)  
**Project Type**: Single-library (benchmark + examples)  
**Performance Goals**: Interactive rendering > ~10 FPS on typical dev laptop; episode setup < 3s; recording overhead acceptable (<2x wall time)  
**Constraints**: Must run with missing moviepy (graceful skip), headless safe; no CLI per user direction  
**Scale/Scope**: Small (single example + helper utilities) — no distributed workloads  

Outstanding clarifications from spec (carry over):
- Multi-scenario sequential mode? (FR-022)
- Frame decimation / sampling? (FR-023)

## Constitution Check
No additional project types introduced (stays within existing single codebase). No external network, no secrets. Complexity low. PASS (initial).

## Project Structure
Docs & design live under `specs/128-classic-interactions-ppo/` plus example script in `examples/` and loader in `robot_sf/benchmark/` (already added). No new package roots needed.

**Structure Decision**: Option 1 (single project) retained.

## Phase 0: Outline & Research
Unknowns / Clarifications:
1. FR-022 multi-scenario chaining needed? Action: default defer; design for trivial extension (list iteration) but not implement now.
2. FR-023 frame sampling needed? Action: defer; add hook stub for future frame decimation.

Research mini-decisions:
- Rendering Source: Use existing environment's internal SimulationView to avoid duplicate frame assembly logic; explicit external SimulationView optional.
- Policy Invocation: Deterministic actions via `predict(..., deterministic=True)` for reproducibility.
- Recording: Reuse env-managed frames if available; fallback to placeholder frames to keep contract stable.
- Headless Mode: Rely on existing environment logic (dummy video driver) rather than bespoke detection.

Decision log (research.md will formalize):
- Decision: Defer multi-scenario; implement single scenario selection with potential extension hook.
- Decision: Expose constants at top of script instead of CLI (per user requirement) ensuring immediate discoverability.
- Decision: Provide summary printouts instead of structured JSON to keep example lightweight.

## Phase 1: Design & Contracts
Entities (refined from spec):
1. ScenarioSelection: name, seeds (ordered), map reference.
2. EpisodeRun: seed, steps, outcome, recorded flag.
3. RunConfig: constants (MODEL_PATH, MATRIX_PATH, SCENARIO_NAME, MAX_EPISODES, RECORD, OUTPUT_DIR, OVERLAY, DRY_RUN).
4. RecordingArtifact (optional future expansion): path, reason (skipped / success).

Data Model (data-model.md to include):
- Provide field table, invariants (non-empty seeds), outcome set: {success, collision, timeout, done}.

Contracts:
- No external API / network endpoints; contracts expressed as Python-level function behaviors.
- Define minimal expectations for `run_demo()` (to be added) returning list[EpisodeRun].

Quickstart (quickstart.md to include):
1. Ensure model file present.
2. Adjust constants optionally.
3. `uv run python examples/classic_interactions_pygame_constants.py` (new script name) runs first scenario.

Test Scenarios (integration):
1. Smoke: DRY_RUN=True -> validates model + scenario file.
2. Seed determinism: first two runs produce identical outcome seeds ordering.
3. Recording skip when moviepy missing (monkeypatch MOVIEPY_AVAILABLE=False) preserves episode loop.

Post-Design Constitution Check: Still compliant (no scope creep). PASS.

## Phase 2: Task Planning Approach (description only)
Task generation will:
- Create tasks for: finalize constants-based script refactor, add `run_demo()` function, add smoke tests, add data-model.md & quickstart.md, implement recording stub improvements, add optional multi-scenario TODO marker.
- Order: data model doc → code refactor → smoke test → recording skip test → documentation linkage → optional enhancements.

## Complexity Tracking
| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|---------------------------------------|
| (none)    |            |                                       |

## Progress Tracking
**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [ ] All NEEDS CLARIFICATION resolved (pending FR-022/FR-023)
- [ ] Complexity deviations documented (none required)

---
*Based on Constitution v1.1.0*
