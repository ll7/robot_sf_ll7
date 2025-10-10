
# Implementation Plan: Accelerate Slow Benchmark Tests

**Branch**: `124-title-accelerate-slow` | **Date**: 2025-09-20 | **Spec**: [/specs/124-title-accelerate-slow/spec.md](./spec.md)
**Input**: Feature specification from `/specs/124-title-accelerate-slow/spec.md`

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
Primary requirement: Enforce a per-test performance budget (soft <20s, hard 60s) without reducing semantic coverage of benchmark tests (resume, reproducibility, orchestration logic). Approach: Define a performance budget policy, minimize heavy scenario matrices, add per-test timeouts, generate slow test report (top 10), provide guidance, and allow relax override.

## Technical Context
**Language/Version**: Python (uv-managed, ≥3.11)  
**Primary Dependencies**: pytest, internal benchmark orchestrator, ruff (indirect)  
**Storage**: N/A (ephemeral test artifacts only)  
**Testing**: pytest (possibly pytest-timeout marker)  
**Target Platform**: macOS dev + Linux CI headless  
**Project Type**: Single library/research framework  
**Performance Goals**: Each test <20s soft; <60s hard; 95% tests under soft limit  
**Constraints**: Preserve semantic assertions; deterministic reduced inputs  
**Scale/Scope**: ~170+ tests; a few heavy benchmark integration tests

## Constitution Check
Alignment:
- Reproducibility preserved (deterministic minimal matrices).
- Benchmark & Metrics unaffected (no schema change).
- Test coverage increases (adds performance assertions) complying with Principle IX.
No contract or schema changes; no violations requiring justification.

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

**Structure Decision**: Option 1 (single project). Optional helper placement under `tests/perf_utils/` (conceptual; final location TBD) for shared minimal matrix logic.

## Phase 0: Outline & Research
Completed. See `research.md` (clarifications resolved: N=10, enforcement policy, relax env var name). No pending unknowns.

## Phase 1: Design & Contracts
Completed. Entities captured in `data-model.md`. No external API endpoints; contracts folder reserved (empty placeholder). Quickstart present. Enforcement uses existing pytest ecosystem; no new public API.

## Phase 2: Task Planning Approach
Tasks (to be generated) will map FR-001..FR-011 to implementation steps:
1. Add/verify per-test timeout markers (<60s) where missing.
2. Create shared minimal scenario matrix helper & reuse across heavy tests.
3. Refactor resume & reproducibility tests to assert minimized episode counts.
4. Integrate timing capture fixture producing Slow Test Report (top 10).
5. Implement guidance generator (episode/horizon/bootstrap reduction suggestions).
6. Add environment variable handling: ROBOT_SF_PERF_RELAX (skip soft fail) and plan for optional ROBOT_SF_PERF_ENFORCE.
7. Update docs (`dev_guide.md`) with performance budget section; link from `docs/README.md`.
8. Add CHANGELOG entry (Unreleased) summarizing policy.
9. Add validation ensuring no metrics or schema regressions (smoke).
10. Add follow-up test verifying guidance output formatting for at least one synthetic slow test.
Ordering: Foundational helpers → test refactors → reporting fixture → documentation & changelog → validation additions.

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
No constitutional violations; section minimized.


## Progress Tracking
**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - description added)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none)

---
*Based on Constitution v1.0.0 - See `/specify/memory/constitution.md`*
