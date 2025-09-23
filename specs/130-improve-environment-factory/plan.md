# Implementation Plan: Improve Environment Factory Ergonomics

**Branch**: `130-improve-environment-factory` | **Date**: 2025-09-22 | **Spec**: `specs/130-improve-environment-factory/spec.md`
**Input**: Feature specification from `/specs/130-improve-environment-factory/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → Found OK
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Collected ambiguities & environment traits
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → No blocking violations (pending clarifications noted)
5. Execute Phase 0 → research.md (to be generated)
6. Execute Phase 1 → contracts/, data-model.md, quickstart.md (design-level for factories)
7. Re-evaluate Constitution Check section post-design
8. Plan Phase 2 task generation approach (describe only; do not create tasks.md)
9. STOP
```

## Summary
Goal: Enhance ergonomics of environment factory functions by making parameter surfaces explicit, adding structured option objects (render/recording), providing deprecation and migration pathways, and strengthening discoverability (docstrings + examples + quickstart). Target: zero breaking runtime behavior; purely additive/improving developer UX while honoring Constitution Principles II (Factory-Based), IV (Unified Config), VII (Backward Compatibility), XII (Logging).

Core approach: Introduce typed `RenderOptions` & `RecordingOptions` dataclasses; update factories to accept these (optional) while retaining primary convenience booleans (`record_video`). Implement a legacy shim capturing previously used kwargs; emit Loguru warnings guiding migration. Extend tests for signature stability, deprecation mapping, invalid combination warnings, and ensure no performance regressions beyond +5% env creation time.

## Technical Context
Language/Version: Python (project currently aligns with 3.13 usage; confirm in runtime env)  
Primary Dependencies: Gymnasium-style interfaces, Loguru, Stable-Baselines3 (optional), NumPy, rendering stack (Pygame / MoviePy)  
Storage: N/A (in-memory objects + optional video artifacts on filesystem)  
Testing: pytest (existing suite: unit + integration + rendering + performance policies)  
Target Platform: Cross-platform, CI headless (macOS + Linux)  
Project Type: Single library/research framework (Option 1 structure)  
Performance Goals: Maintain <1s environment creation; no >5% regression (FR-017)  
Constraints: Backward compatibility (Constitution Principle VII), logging standard (XII), unified config (IV), no direct env class instantiation (II), avoid signature bloat (>~8 primary params)  
Scale/Scope: Limited to ergonomic improvements for 4 factory functions (robot, image, pedestrian, potential multi-robot future placeholder)  

Unknowns / NEEDS CLARIFICATION (from spec):
1. Policy for unknown legacy kwargs: strict error vs permissive mapping.
2. Validation severity for incompatible inputs (warning vs exception).
3. Separation strategy for multi-robot/pedestrian-specific option dataclasses.
4. Mode toggle name for permissive legacy behavior (if chosen).
5. Confirm Python version baseline for dataclass / typing features (e.g., `kw_only`, `Annotated`).

Assumptions (tentative pending clarification):
- Adopt STRICT-by-default (raise) with opt-in permissive via env var `ROBOT_SF_FACTORY_LEGACY=1` for migration (minimizes silent issues).
- Incompatible combos produce WARNING (non-fatal) plus automatic enabling where safe (e.g., auto-enable minimal render pathway for recording).
- Multi-robot path deferred; pedestrian-specific toggles remain inside `RenderOptions` (document future separation plan) to reduce premature complexity.
- Python baseline >=3.11 allows dataclass `slots=True`; will use for lightweight option objects to reduce memory footprint.

## Constitution Check
Mapping to Principles:
- II (Factory Abstraction): Strengthened via explicit signatures and option objects. PASS.
- IV (Unified Configuration): Preserved—config object remains central; new options supplement, not replace. PASS.
- VII (Backward Compatibility): Implement deprecation layer + optional permissive mode; must ensure warnings not errors by default. CONDITIONAL (needs final policy decision).
- VIII (Documentation): Requires updates to `docs/ENVIRONMENT.md`, new migration note folder. PASS contingent on delivery.
- IX (Test Coverage): New tests required (signatures, deprecation warnings, option object behavior). PASS contingent.
- XII (Logging): All new diagnostics via Loguru. PASS.

No constitutional violations provided unknown policies resolved. Complexity minimal (no new architectural layer).  

## Project Structure
Structure Decision: Option 1 (single project) retained. Only spec directory enriched with new artifacts.

### Documentation (this feature)
```
specs/130-improve-environment-factory/
├── spec.md
├── plan.md              # (this file)
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/           # Phase 1 (factory param surface & validation contract)
└── tasks.md             # Phase 2 (NOT created yet)
```

## Phase 0: Outline & Research
Focus Areas:
1. Survey current factory signatures & test coverage (baseline snapshot).
2. Identify all presently used legacy kwargs in repository usage sites (grep search) → build DeprecationMap.
3. Assess typical parameter frequency (which kwargs appear most often) to justify which stay in primary signature.
4. Validate performance cost of additional option object instantiation (micro-benchmark env creation time before/after).
5. Investigate existing patterns in similar RL env libraries (Gymnasium wrappers) for option objects vs flat params (best practices).

Deliverable (`research.md`) Sections:
- Current State Inventory (signatures, call sites).
- Legacy Kwarg Frequency Table.
- Decision Log (unknowns resolved).
- Performance Baseline Numbers (N=30 creations timing).
- Alternatives Considered (builder pattern, dynamic registry, single monolithic options object).

Exit Criteria: All NEEDS CLARIFICATION resolved with decisions & rationale; no unresolved unknowns.

## Phase 1: Design & Contracts
Artifacts:
1. `data-model.md`: Define dataclasses `RenderOptions`, `RecordingOptions`, `DeprecationMap` structure, validation rules.
2. `contracts/`: Provide a markdown contract (or simple schema doc) enumerating factory parameter lists & validation outcomes (pseudo OpenAPI-like table for functions).
3. `quickstart.md`: Before/after examples (old usage -> new usage), minimal recipe for each factory variant, migration snippet.
4. Update agent context (if repository uses AI assistant context file) limited to newly added terminology.

Design Elements:
- Validation Flow Diagram: input params -> normalization -> implicit enabling -> final env creation.
- Logging Points: creation info, deprecation warning, incompatible combo warning, legacy toggle info.
- Deprecation Strategy: 2 release cycle window before strict errors (document timeline expectation; implement warnings only now).

Exit Criteria: Post-design constitution check passes; all FR-001..FR-021 mapped to design components; no unresolved unknowns.

## Phase 2: Task Planning Approach
Task Generation Strategy (future /tasks):
- Each FR becomes at least one implementation + one test task.
- Legacy mapping extraction tasks per legacy kw category.
- Performance validation task (benchmark env creation pre/post, assert delta <5%).
- Documentation tasks: migration guide, environment doc update, examples update.
- Parallelizable tasks flagged: tests, docs, performance benchmark independent of dataclass code once stubs exist.

Ordering:
1. Research finalize (Phase 0)
2. Option dataclasses & contract tests (failing)
3. Deprecation adaptor layer
4. Factory refactor + logging
5. Tests (unit → integration)
6. Docs & examples
7. Performance validation & adjustments

## Phase 3+: Future Implementation
Out of scope for /plan (will execute after tasks.md creation).

## Complexity Tracking
Currently none—feature stays within existing architectural boundaries; no additional project roots or service layers introduced.

## Progress Tracking
Phase Status:
- [ ] Phase 0: Research complete (/plan command)
- [ ] Phase 1: Design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

Gate Status:
- [ ] Initial Constitution Check: PASS (conditionally; finalize after unknowns decisions)
- [ ] Post-Design Constitution Check: PASS
- [ ] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---
*Based on Constitution v1.2.0 - See `.specify/memory/constitution.md`*
# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

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
[Extract from feature spec: primary requirement + technical approach from research]

## Technical Context
**Language/Version**: [e.g., Python 3.11, Swift 5.9, Rust 1.75 or NEEDS CLARIFICATION]  
**Primary Dependencies**: [e.g., FastAPI, UIKit, LLVM or NEEDS CLARIFICATION]  
**Storage**: [if applicable, e.g., PostgreSQL, CoreData, files or N/A]  
**Testing**: [e.g., pytest, XCTest, cargo test or NEEDS CLARIFICATION]  
**Target Platform**: [e.g., Linux server, iOS 15+, WASM or NEEDS CLARIFICATION]
**Project Type**: [single/web/mobile - determines source structure]  
**Performance Goals**: [domain-specific, e.g., 1000 req/s, 10k lines/sec, 60 fps or NEEDS CLARIFICATION]  
**Constraints**: [domain-specific, e.g., <200ms p95, <100MB memory, offline-capable or NEEDS CLARIFICATION]  
**Scale/Scope**: [domain-specific, e.g., 10k users, 1M LOC, 50 screens or NEEDS CLARIFICATION]

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

[Gates determined based on constitution file]

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

**Structure Decision**: [DEFAULT to Option 1 unless Technical Context indicates web/mobile app]

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

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

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
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [ ] Phase 0: Research complete (/plan command)
- [ ] Phase 1: Design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [ ] Initial Constitution Check: PASS
- [ ] Post-Design Constitution Check: PASS
- [ ] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---
*Based on Constitution v1.1.0 - See `/memory/constitution.md`*
