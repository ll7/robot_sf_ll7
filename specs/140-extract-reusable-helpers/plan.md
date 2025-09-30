
# Implementation Plan: Reusable Helper Consolidation

**Branch**: `140-extract-reusable-helpers` | **Date**: 2025-09-30 | **Spec**: [/specs/140-extract-reusable-helpers/spec.md](./spec.md)
**Input**: Feature specification from `/specs/140-extract-reusable-helpers/spec.md`

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
Consolidate reusable helper logic currently scattered across examples and scripts into documented library modules under `robot_sf/`. Maintain behavior parity for all demos while transforming examples/scripts into thin orchestration layers. Deliver a discoverable helper catalog, updated documentation, and regression coverage to ensure extracted helpers remain reliable.

## Technical Context
**Language/Version**: Python 3.11 (per repo toolchain)  
**Primary Dependencies**: Internal `robot_sf` modules, NumPy, Loguru logging, MoviePy (recording), Stable-Baselines3 (policy loading)  
**Storage**: File-based artifacts (configs, JSONL outputs, media); no new persistence required  
**Testing**: Pytest suites (`tests/`, `test_pygame/`), validation scripts under `scripts/validation/`  
**Target Platform**: macOS/Linux development environments with headless CI support  
**Project Type**: single-project Python library + scripts  
**Performance Goals**: Preserve existing simulation throughput (~20–25 steps/sec) and demo startup times (<1s env init)  
**Constraints**: Must honor Constitution Principle XI (helpers in library), maintain Loguru-only logging, avoid breaking environment/benchmark contracts  
**Scale/Scope**: Refactor all maintained demos/examples and high-use scripts; exclude one-off validation/debug scripts per clarification

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I (Reproducible Core)**: Inventory and documentation must preserve deterministic behavior; plan includes regression validation via existing tests and validation scripts.
- **Principle II (Factory Abstraction)**: Extracted helpers will wrap established factory functions without exposing internal env classes; examples remain orchestration layers.
- **Principle VIII (Documentation as API)**: Plan mandates helper catalog docs plus updates to existing guides and example READMEs.
- **Principle IX (Test Coverage)**: Regression tests/validation scripts required for any helper extraction affecting public behavior.
- **Principle XI (Library Reuse & Helper Documentation)**: Core objective—helpers consolidated in `robot_sf/` with docstrings and discoverability.
- **Principle XII (Preferred Logging)**: Refactored helpers continue to use Loguru; plan prohibits introducing new prints in library code.

**Initial Constitution Check**: PASS (requirements incorporated into plan)

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

**Structure Decision**: Option 1 (single project Python library with shared tests)

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

### Phase 0 Deliverables
- Inventory approach for scanning `examples/` and `scripts/` to detect candidate helpers (categorization schema, prioritization rules).
- Guidelines for when helper extraction is justified (e.g., threshold for reuse, complexity, maintenance burden).
- Mapping between helper categories and target library packages (`robot_sf.benchmark`, `robot_sf.render`, `robot_sf.gym_env`, etc.).
- Validation plan listing which automated tests/validation scripts must remain green post-refactor.

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
   - Run `.specify/scripts/bash/update-agent-context.sh copilot`
     **IMPORTANT**: Execute it exactly as specified above. Do not add or remove any arguments.
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, quickstart.md, agent-specific file (no new failing tests required because contracts describe library helper responsibilities rather than HTTP APIs)

### Phase 1 Deliverables
- **data-model.md**: Define `HelperCapability`, `HelperCategory`, `HelperModule`, `ExampleOrchestrator` entities with relationships and metadata (ownership, doc links, test coverage expectations).
- **contracts/helper_catalog.md**: Document the required helper interfaces (e.g., environment setup API, recording utilities, benchmark runners), including input/output expectations and logging requirements.
- **quickstart.md**: Provide step-by-step instructions for consuming the new helper modules from an example script, showing how to migrate an existing demo to the orchestration pattern.
- **Agent context update**: Append summary of new helper modules/locations to `.github/copilot-instructions.md` via the scripted update so other assistants discover them.

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

### Tailoring for This Feature
- Group tasks by helper category (environment setup, recording, benchmarking, misc utilities) to align with data-model entities.
- Include documentation and test update tasks immediately after helper extraction to satisfy Principles VIII & IX.
- Ensure tasks cover de-duplication in each example/script while preserving CI validation coverage.

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
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS (Phase 1 outputs respect Principles I, II, VIII, IX, XI, XII)
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (N/A)

---
*Based on Constitution v1.3.0 - See `/memory/constitution.md`*
