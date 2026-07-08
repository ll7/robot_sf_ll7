
# Implementation Plan: Preserve Algorithm Separation in Benchmark Aggregation

**Branch**: `142-aggregation-mixes-algorithms` | **Date**: 2025-10-06 | **Spec**: [Feature Spec](../142-aggregation-mixes-algorithms/spec.md)
**Input**: Feature specification from `/specs/142-aggregation-mixes-algorithms/spec.md`

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
The classic benchmark currently groups episode metrics by `scenario_params.algo`, but the orchestrator only records the algorithm name at the top level. This plan ensures every episode carries the algorithm identifier under both `algo` and `scenario_params["algo"]`, teaches the aggregators to fall back to the top-level field, and adds proactive warnings when expected baselines are missing so analysts always see per-algorithm metrics.

## Technical Context
**Language/Version**: Python ≥3.11 (repo targets 3.11–3.12 via uv + Ruff)  
**Primary Dependencies**: `robot_sf.benchmark` modules, Loguru logging, NumPy, JSON/JSONSchema utilities, pytest suite  
**Storage**: JSONL/JSON files under `results/` (append-only episode logs and aggregates)  
**Testing**: Pytest suites in `tests/` plus benchmark smoke validations (`scripts/validation/*.py`)  
**Target Platform**: Headless macOS/Linux runners (CI + local)  
**Project Type**: single-project Python library/application  
**Performance Goals**: Maintain ability to aggregate ≥10k episodes without exceeding existing compute_aggregates runtime budgets (<2s per 5k episodes on dev hardware)  
**Constraints**: Preserve JSON schema compatibility (Principle VII), use Loguru for diagnostics, fail fast on missing metadata, warn-but-continue on missing algorithms  
**Scale/Scope**: Classic benchmark matrix (≈3 algorithms × ~10 scenarios × ≤10 episodes each) with potential expansion to larger episode sets

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle III (Benchmark & Metrics First)**: Plan maintains structured JSONL outputs and strengthens grouping metadata; passes.  
- **Principle VI (Metrics Transparency)**: Aggregation changes continue to expose per-algorithm statistics with warnings for missing data; passes.  
- **Principle VII (Backward Compatibility)**: Mirroring `algo` under `scenario_params` preserves schema expectations while leaving top-level field intact; no breaking change anticipated.  
- **Principle XI (Library Reuse & Helper Documentation)**: Changes will live in existing benchmark library modules with docstrings and logging updates as needed; compliant.  
- **Principle XII (Preferred Logging & Observability)**: Warning path will use Loguru warning level rather than prints; compliant.

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

**Structure Decision**: Option 1 (single Python project); no frontend/mobile splitting required.

## Phase 0: Outline & Research
1. Confirm where algorithm metadata is produced (`run_full_benchmark` manifests, orchestrator payloads) and identify hook to mirror `algo` into `scenario_params`.  
2. Review `compute_aggregates_with_ci` fallback logic and determine minimal changes needed to honour top-level `algo` while preserving dotted-path configuration.  
3. Survey existing validation scripts and docs to capture required updates (documentation entry, changelog notes).  
4. Capture decisions plus alternatives (e.g., rewriting aggregation default vs. enriching records) in `research.md` following Decision/Rationale/Alternatives structure.

**Output**: `/specs/142-aggregation-mixes-algorithms/research.md` documenting the above findings.

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. Document the episode record and aggregation summary structures in `data-model.md`, including new `scenario_params.algo` mirroring, warning annotations, and summary metadata for missing algorithms.  
2. Define library-level contract expectations in `/specs/142-aggregation-mixes-algorithms/contracts/` (e.g., `compute_aggregates_with_ci` grouping behaviour, orchestration metadata injection) with request/response examples for programmatic consumers.  
3. Outline validation-focused quickstart steps in `quickstart.md` (run benchmark slice, trigger aggregation, inspect warnings) aligned with smoke tests.  
4. Identify failing tests to author (e.g., new pytest cases verifying grouping and warnings) and capture them as part of contracts/quickstart references.  
5. Update GitHub Copilot agent context via `.specify/scripts/bash/update-agent-context.sh copilot`, adding only the new libraries/concepts introduced in this plan.

**Output**: `/specs/142-aggregation-mixes-algorithms/data-model.md`, `/specs/142-aggregation-mixes-algorithms/contracts/*`, `/specs/142-aggregation-mixes-algorithms/quickstart.md`, updated agent context note.

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as the scaffold.
- Derive contract-test tasks from `/contracts/aggregation.md` and `/contracts/orchestrator.md` (pytests covering metadata injection, fallback behaviour, warning emission, error cases). [P]
- Map `data-model.md` entities to implementation tasks (orchestrator writer update, aggregation fallback/path metadata, `_meta` diagnostics). [P]
- Translate `quickstart.md` flow into integration validation tasks (smoke benchmark run, manual aggregation warning scenario, documentation updates).
- Add documentation + changelog tasks mandated by spec (docs/benchmark.md, CHANGELOG.md) and validation script updates.
- Include clean-up tasks for legacy JSONL detection and custom exception plumbing.

**Ordering Strategy**:
- TDD-centred: author regression tests (contract + integration) before modifying production code.
- Dependency aware: orchestrator metadata injection (source of truth) precedes aggregator fallback implementation; docs and changelog follow functional changes.
- Mark independent documentation tasks and changelog updates as [P] for parallel execution once code/tests stabilize.

**Estimated Output**: ~18-22 ordered tasks capturing tests, implementation, docs, and verification.

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
| _None_ | – | – |


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
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented

---
*Based on Constitution v1.3.0 - See `/memory/constitution.md`*
