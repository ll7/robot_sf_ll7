
# Implementation Plan: Fix Benchmark Placeholder Outputs

**Branch**: `133-all-generated-plots` | **Date**: 2025-09-24 | **Spec**: /specs/133-all-generated-plots/spec.md
**Input**: Feature specification from `/specs/133-all-ge**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed-plots/spec.md`

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
Fix benchmark placeholder outputs by implementing real visualization generation. Replace dummy plots and videos with actual statistical plots of metrics data and rendered simulation replays. Add validation to ensure outputs contain real data rather than placeholders. Integrate visualization generation into benchmark pipeline while maintaining backward compatibility.

## Technical Context
**Language/Version**: Python 3.13  
**Primary Dependencies**: Matplotlib, MoviePy, NumPy, Gymnasium, StableBaselines3  
**Storage**: JSONL episode data, YAML scenario configs, PDF plots, MP4 videos  
**Testing**: pytest, headless GUI tests  
**Target Platform**: Linux/macOS with optional GPU support  
**Project Type**: single (Python research framework)  
**Performance Goals**: Benchmark execution < 5 minutes, video rendering < 30 seconds per scenario  
**Constraints**: Must work headless (no display), reproducible outputs, vector PDF plots  
**Scale/Scope**: 10-100 episodes per benchmark run, multiple baselines, statistical analysis

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Principle I (Reproducible Research Core)**: ✅ PASS - This feature improves reproducibility by ensuring benchmark outputs are real rather than placeholders, enabling proper analysis of social navigation research.

**Principle II (Factory-Based Environment)**: ✅ PASS - Feature uses existing factory functions and doesn't change environment interfaces.

**Principle III (Benchmark & Metrics First)**: ✅ PASS - Directly addresses benchmark output quality and metric visualization.

**Principle IV (Unified Configuration)**: ✅ PASS - Uses existing config system, no changes needed.

**Principle V (Minimal Baselines)**: ✅ PASS - Doesn't add new baselines, improves existing benchmark functionality.

**Principle VI (Metrics Transparency)**: ✅ PASS - Ensures metrics are properly computed and visualized.

**Principle VII (Backward Compatibility)**: ✅ PASS - No breaking changes to public APIs.

**Principle VIII (Documentation as API)**: ✅ PASS - Feature improves documentation through better visual outputs.

**Principle IX (Test Coverage)**: ✅ PASS - Will add tests for plot/video generation.

**Principle X (Scope Discipline)**: ✅ PASS - Strictly within social navigation benchmark scope.

**Principle XI (Internal Maintainability)**: ✅ PASS - Improves code quality by fixing placeholder implementations.

**Principle XII (Logging & Observability)**: ✅ PASS - No changes to logging requirements.

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
   - Run `.specify/scripts/bash/update-agent-context.sh copilot`
     **IMPORTANT**: Execute it exactly as specified above. Do not add or remove any arguments.
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
- Each contract function → implementation task [P]
- Each data entity → validation task
- Each user scenario → integration test task
- Error handling and dependency checking tasks

**Ordering Strategy**:
- TDD order: Contract tests before implementation
- Dependency order: Core visualization functions before integration
- Parallel execution: Independent plotting/video tasks can run concurrently [P]
- Sequential for integration: Benchmark orchestrator changes depend on visualization functions

**Estimated Output**: 15-20 numbered, ordered tasks in tasks.md

**Key Task Categories**:
1. **Core Implementation** [P]: `generate_benchmark_plots()`, `generate_benchmark_videos()`
2. **Data Processing**: Episode parsing, metric extraction, trajectory handling
3. **Integration**: Extend benchmark orchestrator, add validation
4. **Error Handling**: Dependency checks, graceful failures, user feedback
5. **Testing**: Contract tests, integration tests, validation tests
6. **Documentation**: Update existing docs, add troubleshooting

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
- [ ] Complexity deviations documented

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
