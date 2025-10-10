
# Implementation Plan: Verify Feature Extractor Training Flow

**Branch**: `141-check-that-the` | **Date**: 2025-10-02 | **Spec**: [spec](./spec.md)
**Input**: Feature specification from `/Users/lennart/git/robot_sf_ll7/specs/141-check-that-the/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Enhance `scripts/multi_extractor_training.py` so that each registered feature extractor can be validated end-to-end on macOS (single-thread spawn mode) and Ubuntu RTX (vectorized workers), while writing timestamped outputs under `./tmp/multi_extractor_training/`, emitting both JSON and Markdown summaries, and consolidating reusable helper logic inside `robot_sf/` for cross-script reuse.

## Technical Context
**Language/Version**: Python ≥3.11 (uv-managed virtual environment)  
**Primary Dependencies**: Stable-Baselines3, Gymnasium, Torch, Loguru, JSONSchema  
**Storage**: Local filesystem (timestamped directories under `./tmp/multi_extractor_training/`)  
**Testing**: pytest with headless environment smoke tests  
**Target Platform**: macOS 15 (Apple Silicon M4) and Ubuntu 22.04 with NVIDIA RTX GPUs  
**Project Type**: single  
**Performance Goals**: Default macOS run completes without crash; Ubuntu vectorized mode sustains multi-env execution without schema drift; summaries generated within seconds post-run  
**Constraints**: Enforce spawn start method on macOS, honor Constitution Principle XI by moving reusable helpers into `robot_sf/`, ensure logging via Loguru with clear severity separation, avoid interactive prompts, preserve prior tmp artifacts  
**Scale/Scope**: Typical comparison of 3–6 extractors per run; aggregated summaries expected to remain under 1 MB

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I (Reproducibility)**: Timestamped output directories and structured summaries guarantee reruns remain auditable.  
- **Principle II (Factory Abstraction)**: Training script will continue to create environments via `environment_factory.make_*`, avoiding direct instantiation.  
- **Principle III (Benchmark & Metrics)**: JSON summary aligns with existing metrics pipeline and remains append-only.  
- **Principle XI (Library Reuse & Helper Documentation)**: Shared logic for extractor registry, hardware capture, and summary writing will live under `robot_sf/` with docstrings.  
- **Principle XII (Logging)**: All runtime messaging will use Loguru with warning/error separation.

**Status**: PASS (initial and post-design review)

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure]
```

**Structure Decision**: Option 1 (single project)

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - Output retention policy → resolved via timestamped directory research.
   - Cross-platform worker configuration → documented spawn vs. vectorized strategies.
   - Summary artifact format → JSON + Markdown decision.

2. **Generate and dispatch research agents**:
   - Captured decisions in `research.md` with rationale and rejected alternatives.

3. **Consolidate findings** in `research.md` (completed in this command).

**Output**: [`research.md`](./research.md)

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → Documented in [`data-model.md`](./data-model.md) covering `ExtractorConfigurationProfile`, `ExtractorRunRecord`, `HardwareProfile`, and `TrainingRunSummary` with validation and relationships.

2. **Generate contracts** from functional requirements:
   - Authored JSON schema for `summary.json` in [`contracts/training_summary.schema.json`](./contracts/training_summary.schema.json).
   - Defined Markdown summary structure in [`contracts/summary_markdown.md`](./contracts/summary_markdown.md).

3. **Plan contract tests**:
   - Future pytest modules will validate JSON files against the schema and assert Markdown sections exist (to be generated during /tasks).

4. **Extract test scenarios**:
   - Quickstart outlines macOS single-thread and Ubuntu GPU flows (see [`quickstart.md`](./quickstart.md)).

5. **Update agent file incrementally**:
   - Triggered `.specify/scripts/bash/update-agent-context.sh copilot` after drafting plan artifacts; script updated `.github/copilot-instructions.md` with latest context.

**Output**: [`data-model.md`](./data-model.md), [`contracts/`](./contracts), [`quickstart.md`](./quickstart.md)

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each contract → contract test task [P]
- Each entity → model creation task [P] 
- Each user story → integration test task
- Implementation tasks to make tests pass

**Ordering Strategy**:
- TDD order: Tests before implementation 
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| _None_ | — | — |


## Progress Tracking
*This checklist is updated during execution flow*

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
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented

---
*Based on Constitution v1.3.0 - See `/memory/constitution.md`*
