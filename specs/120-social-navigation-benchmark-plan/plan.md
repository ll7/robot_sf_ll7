
# Implementation Plan: Social Navigation Benchmark Platform Foundations (Extended: Classical Interaction Maps Pack)

**Branch**: `120-social-navigation-benchmark-plan` | **Date**: 2025-09-19 | **Spec**: `specs/120-social-navigation-benchmark-plan/spec.md`
**Input**: Feature specification plus extension scope: add canonical robot–pedestrian interaction SVG maps & scenario config.

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
Core benchmark foundations implemented. This extension adds a curated set of SVG maps modeling classical human–robot interaction patterns (crossing, head‑on corridor passing, overtaking, bottleneck negotiation, doorway traversal, merging flows, T‑intersection, group crossing). A unified scenario matrix (`classic_interactions.yaml`) will reference these maps with density/flow variants, enabling reproducible evaluation of social navigation policies in well‑studied micro‑navigation motifs.

## Technical Context
**Language/Version**: Python 3.12+ (uv managed)  
**Primary Dependencies**: Gymnasium-like env layer (internal), Fast-pysf (pedestrian physics submodule), NumPy, Matplotlib (figures), PPO model artifacts.  
**Storage**: File-system (JSONL episodes, SVG maps, generated figures).  
**Testing**: pytest (unit + integration + schema), headless CI with dummy video backend.  
**Target Platform**: Linux (CI) & macOS dev; headless execution required.  
**Project Type**: Single Python library + scripts (benchmark & docs generators).  
**Performance Goals**: Maintain creation <3s, reset throughput soft >0.5/s (already instrumented).  
**Constraints**: Deterministic seeding; maps must not introduce non-deterministic ordering; keep SVG obstacle sets minimal (perf).  
**Scale/Scope**: 8 new interaction map archetypes * density variants (low/med/high) generating ~24–32 scenario entries.

## Constitution Check
All additions comply with Constitution principles:
I Reproducibility: Maps are deterministic static SVG assets (versioned).  
II Factory Abstraction: No new factory functions; scenarios reference existing env factory.  
III Benchmark First: Each scenario produces standardized episodes (no schema change).  
IV Unified Config: Scenario YAML only; no ad-hoc kwargs added in code.  
V Minimal Baselines: No new baselines introduced.  
VI Metrics Transparency: Metrics unchanged; scenarios just expand coverage.  
VII Backward Compatibility: No interface or schema change → no version bump.  
VIII Documentation: New quickstart + index link will be added.  
IX Tests: Add smoke for scenario matrix validation + one spawn parsing test.  
X Scope Discipline: Focus remains social navigation micro‑interaction contexts.

Initial Gate: PASS (no violations requiring Complexity Tracking).

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

**Structure Decision**: Option 1 (single project) — existing repository layout already conforms.

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

**Output**: research.md (classical interaction taxonomy + design decisions) — TO BE GENERATED (this run will create file).

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

Adaptation: No external API endpoints; contracts map to internal scenario schema validation. A lightweight contract document (schema reference table) will be included in `data-model.md`. No new failing tests auto-generated here; instead we will plan explicit pytest additions in Phase 2 tasks (matrix validation + spawn zone parsing). Quickstart will document invoking the new matrix with benchmark runner.

**Output (for this extension)**: data-model.md (map archetype table + scenario matrix spec), quickstart.md (usage for classic interactions), research.md.

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
- [x] Initial Constitution Check: PASS
- [ ] Post-Design Constitution Check: PASS
- [ ] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
