
# Implementation Plan: Integrate Plots & Videos into Full Classic Benchmark

**Branch**: `126-title-integrate-plots` | **Date**: 2025-09-20 | **Spec**: `specs/126-title-integrate-plots/spec.md`
**Input**: Feature specification from `specs/126-title-integrate-plots/spec.md`

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
Wire existing placeholder plot & video generation into the Full Classic Benchmark orchestrator so that after the adaptive benchmark loop finishes a single post-processing pass produces (a) deterministic lightweight plots under `plots/` and (b) representative episode videos under `videos/` accompanied by machine-readable manifests (`plot_artifacts.json`, `video_artifacts.json`). Primary renderer for videos is the existing headless-capable PyGame `SimulationView` replaying recorded episode state; fallback is the current synthetic path renderer when `SimulationView` cannot initialize or replay state. No changes to existing episode or aggregate JSON schemas; optional dependencies (matplotlib, moviepy/ffmpeg, pygame) degrade gracefully via skipped artifact entries. Performance soft budgets: plots < 2s, one default video < 5s (documented if exceeded, not failing run). Deterministic selection: first N completed episodes (respect `max_videos`).

## Technical Context
**Language/Version**: Python 3.11 (repo standard)  
**Primary Dependencies**: Core repo (robot_sf), optional: matplotlib (plots), pygame (`SimulationView`), moviepy + ffmpeg (mp4 encoding; fallback synthetic writer already present).  
**Storage**: File system outputs (JSONL episodes, JSON summaries, PDFs, MP4s).  
**Testing**: pytest (unit + integration); add new tests under `tests/benchmark_full_classic/` or reuse existing structure; headless video tests require dummy SDL vars.  
**Target Platform**: Headless CI (Linux/macOS) + local dev.  
**Project Type**: Single library/research benchmark repository (Option 1 structure).  
**Performance Goals**: Additional overhead: plots < 2s total, one video < 5s (soft), deterministic selection; zero impact on adaptive convergence logic.  
**Constraints**: Must not break benchmark resume semantics; no new CLI flags unless later justified; no schema changes to existing JSON files; graceful skip on missing optional deps; reproducible outputs (stable filenames).  
**Scale/Scope**: Typical benchmark run: tens of episodes (<= ~200) — video generation limited to small N (default small, user bounded by existing `max_videos`).

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Assessment vs Constitution (v1.0.0):
- Principle I (Reproducibility): Deterministic episode selection & stable filenames uphold reproducibility.
- Principle II (Factory Abstraction): No change to env factories; compliant.
- Principle III (Benchmark & Metrics First): Adds non-metric artifacts post-run without altering metrics pipeline; compliant.
- Principle IV (Unified Config): Reuses existing flags; no new ad-hoc params; compliant.
- Principle V (Baselines): Unaffected.
- Principle VI (Transparency): Manifests with status/notes increase transparency.
- Principle VII (Backward Compatibility): No schema breaks; compliant.
- Principle VIII (Documentation): Spec + quickstart ensure discoverability; will update docs index on implementation PR.
- Principle IX (Test Coverage): Plan includes new tests for artifact creation & skip logic.
- Principle X (Scope Discipline): Feature squarely within visualization/analysis; in scope.

Result: PASS (no violations). Post-design recheck unchanged.

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

**Structure Decision**: Option 1 (single project) retained — only adds logic inside existing benchmark module and new tests.

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved (Completed; no remaining unknowns).

## Phase 1: Design & Contracts
*Prerequisites: research.md complete (Yes)*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh copilot` for your AI assistant
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

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
No violations; table intentionally omitted.


## Progress Tracking
**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (tasks.md created)
- [x] Phase 3: Tasks generated (tasks.md populated)
- [x] Phase 4: Implementation complete (core feature delivered; deferred items noted in tasks.md)
- [x] Phase 5: Validation passed (tests green; quality gates run; performance within soft budgets)

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none required)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
