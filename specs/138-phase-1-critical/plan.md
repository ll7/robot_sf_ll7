
# Implementation Plan: Type Checking Fixes

**Branch**: `138-phase-1-critical` | **Date**: September 26, 2025 | **Spec**: /Users/lennart/git/robot_sf_ll7/specs/138-phase-1-critical/spec.md
**Input**: Feature specification from `/specs/138-phase-1-critical/spec.md`

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
Resolve 103 type checking diagnostics in the robot_sf codebase through a phased approach: Phase 1 fixes critical compatibility and runtime issues, Phase 2 updates type annotations, Phase 3 handles import resolution, and Phase 4 improves overall type safety. Technical approach involves analyzing current diagnostics, implementing fixes while maintaining backward compatibility, and achieving 80% type coverage validated with uvx ty type checker.

## Technical Context
**Language/Version**: Python 3.11 and later  
**Primary Dependencies**: uvx ty (mypy-based type checker), Python standard library  
**Storage**: N/A  
**Testing**: pytest  
**Target Platform**: Any Python 3.11+ compatible platform  
**Project Type**: single (codebase quality improvement)  
**Performance Goals**: N/A (type checking is static analysis)  
**Constraints**: Maintain backward compatibility, no breaking changes to public APIs  
**Scale/Scope**: Resolve 103 type diagnostics across ~15k lines of Python code

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle XI (Internal Maintainability)**: ✅ COMPLIES - Type annotations and safety improvements enhance long-term maintainability and reproducibility
- **Quality Gates**: ✅ COMPLIES - Type analysis with zero unreviewed errors is a required gate
- **Development Workflow**: ✅ COMPLIES - Changes to public contracts avoided (backward compatibility maintained)
- **Scope Discipline**: ✅ COMPLIES - Focused on social navigation codebase type safety, no out-of-scope additions
- **No violations detected** - Implementation can proceed without complexity justifications

*Post-Design Check: All requirements still compliant with constitution principles*

## Project Structure

### Documentation (this feature)
```
specs/138-phase-1-critical/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT - APPLIES)
robot_sf/
├── gym_env/
├── sim/
├── benchmark/
└── [other modules with type fixes]

tests/
└── [type-related test updates]

# No changes to overall structure - type fixes applied in-place
```

**Structure Decision**: Option 1 (single project) - Type fixes applied in-place to existing codebase structure without architectural changes

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - Current type diagnostics breakdown (103 issues categorized)
   - Specific files and modules affected
   - Python version compatibility requirements
   - uvx ty configuration and options

2. **Generate and dispatch research agents**:
   ```
   Task: "Analyze current uvx ty check . --exit-zero output to categorize the 103 diagnostics"
   Task: "Research Python 3.11+ datetime.UTC import patterns and compatibility"
   Task: "Identify factory function signatures requiring type annotation updates"
   Task: "Review Gym space type issues in reinforcement learning contexts"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all unknowns resolved

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
- Each diagnostic category → implementation task [P]
- Each entity type → validation task
- Each phase requirement → verification task
- Implementation tasks to resolve diagnostics systematically

**Ordering Strategy**:
- Phase order: Phase 1 (critical) → Phase 2 → Phase 3 → Phase 4
- Within phases: High-impact fixes first (runtime errors before annotations)
- Mark [P] for parallel execution (independent file fixes)

**Estimated Output**: 15-20 numbered, ordered tasks in tasks.md

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
*Based on Constitution v1.2.0 - See `/memory/constitution.md`*
