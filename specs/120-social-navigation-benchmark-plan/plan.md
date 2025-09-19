
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

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Compliance (Initial) | Notes |
|-----------|----------------------|-------|
| I Reproducibility | PASS | Deterministic seeds + manifest resume + provenance hashes planned. |
| II Factory Abstraction | PASS | Reuses existing factory env creation only; no direct class instantiation. |
| III Benchmark & Metrics First | PASS | Metrics + JSONL schema central; no hidden state outputs. |
| IV Unified Config | PASS | Scenario matrix + unified config objects; no ad-hoc kwargs. |
| V Baselines Minimal | PASS | Limiting to SocialForce, PPO, Random (ORCA deferred). |
| VI Metrics Transparency | PASS | Provide raw metrics + SNQI decomposition + optional CIs. |
| VII Backward Compatibility | PASS | Additive schema v1; version field included; no breaking factory changes. |
| VIII Documentation Surface | PASS | Will add benchmark docs + SNQI weight tooling references in docs index. |
| IX Test Coverage | PASS | Plan unit tests for metrics, smoke tests for baselines, resume tests. |
| X Scope Discipline | PASS | Excludes dashboards, unrelated algorithms, multi-robot features. |

No violations requiring complexity table entries at this stage.

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

**Output**: research.md with all NEEDS CLARIFICATION resolved

Phase 0 topics identified:
1. SNQI normalization formal definition boundary (percentile vs fixed baseline) → finalize doc language.
2. Bootstrap sampling defaults (samples=1000? confidence=0.95) → justify trade-off performance vs statistical stability.
3. Episode identity hashing scheme (fields included) → ensure stability without including volatile timing.
4. Collision threshold standardization (distance constant) → cite source or rationale.
5. Force comfort threshold value justification (link to literature or internal heuristic).
6. Pareto frontier metric pair selection (which pairs canonical) → document rationale.
7. Resume manifest invalidation triggers (file size change vs hash) → specify algorithm.
8. SNQI weight provenance fields (which metadata stored) → define list.

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

Initial entities (to be captured in data-model.md):
- ScenarioSpec (id, density, flow_pattern, obstacles, group_behavior_flags, repetitions)
- EpisodeRecord (episode_id, scenario_id, seed, algo_id, metrics, timings, status, provenance)
- Metrics (primitive scalars + structured subgroups: distances, forces, smoothness)
- SNQIWeights (version, components, weights, baseline_stats_hash)
- AggregateSummary (group_key, metric_stats, ci_bounds?)
- ResumeManifest (episodes_index_hash, count, file_size, schema_version, updated_at)

Contracts to draft (files under contracts/):
1. `episode.schema.v1.json` — JSON Schema for episode record.
2. `aggregate.schema.v1.json` — Summary output schema.
3. `scenario-matrix.schema.v1.json` — Scenario matrix definition.
4. `snqi-weights.schema.v1.json` — Weight artifact schema.
5. `resume-manifest.schema.v1.json` — Manifest sidecar schema.

Quickstart key steps (to appear in quickstart.md):
1. Validate scenario matrix.
2. Run benchmark to produce episodes JSONL.
3. Compute baseline stats and SNQI weights.
4. Aggregate with bootstrap CIs.
5. Generate figures + tables.
6. Reproduce SNQI ablation results.

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
*Based on Constitution v1.0.0 - See `.specify/memory/constitution.md`*
