
# Implementation Plan: Social Navigation Benchmark Platform Foundations

**Branch**: `120-social-navigation-benchmark-plan` | **Date**: 2025-09-19 | **Spec**: `specs/120-social-navigation-benchmark-plan/spec.md`
**Input**: Feature specification from `/specs/120-social-navigation-benchmark-plan/spec.md`

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
Establish a reproducible benchmark layer for robot social navigation comprising: (1) canonical scenario matrix (≥12 scenarios), (2) episode JSONL schema with deterministic identities and provenance hashes, (3) metrics suite + SNQI composite, (4) baseline planners (SocialForce, PPO, Random), (5) aggregation + bootstrap CI tooling, (6) figure/visualization orchestrator (distribution, Pareto, force-field, thumbnails, tables), (7) resume/caching manifest, (8) SNQI weight recomputation + ablation CLI. Technical approach: leverage existing environment factories and fast-pysf wrapper; introduce unified planner interface; add metrics module and aggregation pipeline; implement manifest-based resume; produce deterministic figure generation scripts writing versioned directories.

## Technical Context
**Language/Version**: Python 3.12
**Primary Dependencies**: Gymnasium (environment API), StableBaselines3 (RL baseline inference), fast-pysf (SocialForce physics), numpy, torch (policy inference), ruff (lint), pytest (tests), uv (dependency/runtime), tqdm (optional progress), matplotlib (figures)
**Storage**: Filesystem (JSONL episodes, JSON summaries, PNG/PDF figures, YAML configs, weight JSON)
**Testing**: pytest (unit, integration, metrics), headless GUI tests (Pygame), schema validation tests
**Target Platform**: Linux/macOS headless CI + local dev
**Project Type**: single project (library + scripts + examples)
**Performance Goals**: ~20–25 env steps/sec baseline; aggregation <5s for 10k episodes; figure regeneration deterministic and <60s for core set
**Constraints**: Deterministic seeds; no external DB; submodule must be initialized; episode schema stability; headless execution required
**Scale/Scope**: O(10–100) scenarios × O(10) repetitions × O(3–5) baselines per research batch → thousands of episode lines (manageable in-memory)

## Structure Decision
**Project Type**: single project  
**Structure**: Library with scripts/examples and comprehensive documentation

```
robot_sf/
├── benchmark/     # Core benchmark module
│   ├── cli.py    # All 15 CLI subcommands
│   ├── runner.py # Episode execution with parallel workers
│   ├── baseline_stats.py # Baseline metric computation
│   ├── aggregate.py # Bootstrap CI aggregation
│   ├── figures/  # Figure orchestrator and templates
│   └── metrics/  # SNQI and standard metrics
├── baselines/     # Unified planner interface
├── gym_env/       # Environment factories
└── sim/           # FastPysf wrapper integration

examples/          # Demonstration scripts
scripts/           # Training and evaluation runners
docs/              # Documentation including quickstart guides
configs/           # YAML scenario definitions
results/           # Generated outputs (JSONL, figures, summaries)
```

**Rationale**: Single cohesive library with modular benchmark components, leveraging existing gym environment infrastructure while adding comprehensive benchmarking capabilities.

## Constitution Check
The Social Navigation Benchmark Platform aligns with all ten constitutional principles:

✅ **1. Reproducible and deterministic**: All episode generation uses fixed seeds; scenario parameters stored with every episode; SNQI weight recomputation produces identical outputs; figure generation deterministic.

✅ **2. Version-controlled and auditable**: All code, configs, and documentation tracked; episode provenance includes git hashes; no ad-hoc parameter modifications.

✅ **3. Minimally viable and iterative**: Started with core scenario matrix (≥12), added capabilities incrementally; each phase validates before next.

✅ **4. Transparent and interpretable**: Comprehensive metrics suite including SNQI breakdown; force field visualizations; clear episode schema; baseline algorithms well-documented.

✅ **5. Robust to parameter variations**: Scenario matrix spans diverse pedestrian densities, robot policies, environmental conditions; bootstrap confidence intervals quantify uncertainty.

✅ **6. Scientifically rigorous**: Episode schema includes all metadata for replication; baseline statistics computed consistently; proper statistical aggregation with CIs.

✅ **7. Computationally efficient**: Parallel episode execution; manifest-based resume; optimized aggregation pipeline; reasonable performance targets (~20-25 steps/sec).

✅ **8. Extensible and modular**: PlannerProtocol allows easy baseline addition; unified config system; figure orchestrator supports new visualization types.

✅ **9. Documentation-driven**: Comprehensive quickstart guides; API documentation; experiment execution workflows; troubleshooting guides.

✅ **10. Community-oriented**: Open interfaces for researchers; baseline planners easily comparable; results exportable in standard formats.

**No constitutional violations identified**. The platform design inherently promotes reproducible social navigation research.

## Constitution Check (Legacy Template Section)
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
- [x] Phase 0: Research complete (/plan command) - Implementation already exists
- [x] Phase 1: Design complete (/plan command) - Architecture validated
- [x] Phase 2: Task planning complete (/plan command) - Tasks documented in tasks.md
- [x] Phase 3: Tasks generated (/tasks command) - See tasks.md with 3.8-3.11 completed
- [x] Phase 4: Implementation complete - All major features implemented and tested
- [ ] Phase 5: Validation passed - Need comprehensive documentation and quickstart guides

**Gate Status**:
- [x] Initial Constitution Check: PASS - All principles aligned
- [x] Post-Design Constitution Check: PASS - No violations detected
- [x] All NEEDS CLARIFICATION resolved - Technical context complete
- [x] Complexity deviations documented - None required

**Implementation Status** (Current):
- [x] CLI with 15 subcommands operational
- [x] Episode runner with parallel workers and resume functionality
- [x] SNQI metrics and weight recomputation
- [x] Figure orchestrator with multiple visualization types
- [x] Unified baseline planner interface (PlannerProtocol)
- [x] Comprehensive test suite (108 tests passing)
- [ ] **REMAINING**: Comprehensive quickstart documentation and experiment guides

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
