# Implementation Plan: Episode Video Artifacts MVP

**Branch**: `132-episode-video-artifacts` | **Date**: 2025-09-24 | **Spec**: /Users/lennart/git/robot_sf_ll7/specs/132-episode-video-artifacts/spec.md
**Input**: Feature specification from `/specs/132-episode-video-artifacts/spec.md`

## Execution Flow (/plan command scope)

1. Load feature spec from Input path — OK
2. Fill Technical Context (scan for NEEDS CLARIFICATION) — Clarifications provided in spec, Session 2025-09-24
3. Fill the Constitution Check section based on the constitution document — below
4. Evaluate Constitution Check section — PASS (no violations)
5. Execute Phase 0 → research.md — created and summarized
6. Execute Phase 1 → contracts, data-model.md, quickstart.md — created stubs
7. Re-evaluate Constitution Check — PASS
8. Plan Phase 2 → Describe task generation approach — documented
9. STOP - Ready for /tasks command

## Summary
From the spec: provide optional per-episode MP4 artifacts with a no-video toggle, deterministic renderer selection (synthetic by default, SimulationView on request), manifest metadata in-episode, micro-batch integration test, and <5% overhead budget. Documentation will be linked from docs index and benchmark TODO.

## Technical Context
**Language/Version**: Python 3.13 (repo setup via uv)  
**Primary Dependencies**: MoviePy (optional), NumPy, Pygame (SimulationView path), Loguru (logging)  
**Storage**: Filesystem (results/videos/ under run output stem)  
**Testing**: pytest (existing suite), micro-batch integration test to be added  
**Target Platform**: macOS + headless CI  
**Project Type**: single  
**Performance Goals**: Video overhead < 5% added wall time (soft warn; enforce via env)  
**Constraints**: Headless resilience; deterministic naming; resume-safe; no schema breaks without versioning  
**Scale/Scope**: Dozens to hundreds of episodes per batch; videos optional

## Constitution Check
- Principle I (Reproducibility): Deterministic naming and in-episode manifest maintain traceability — compliant.
- Principle II (Factory-based): No direct env constructors added — compliant.
- Principle III (Benchmark-first): JSONL manifest fields preserved; schema extension kept explicit — compliant.
- Principle VII (Backward Compatibility): No breaking changes to public factories; new flags additive — compliant.
- Principle XII (Logging): Use Loguru for warnings/errors (skip reasons) — compliant.

Result: PASS (Initial Constitution Check)

## Project Structure

### Documentation (this feature)
```
specs/132-episode-video-artifacts/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Created by /tasks
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
```

**Structure Decision**: DEFAULT (single project)

## Phase 0: Outline & Research
- Unknowns resolved by clarifications:
  - CLI/API flags, defaults, renderer selector, output layout, manifest fields, performance policy, headless behavior, test scope.
- Best practices to note:
  - Use synthetic renderer as default to avoid backend fragility.
  - In-episode JSONL artifact metadata simplifies aggregation and resume.
  - Soft budget enforcement via env var avoids flakiness in CI.

Output written to research.md summarizing decisions and rationale.

## Phase 1: Design & Contracts
- Entities (data-model.md): Episode Video Artifact, Renderer Mode, Performance Budget.
- Contracts: Document CLI flags and expected manifest JSON shape in contracts/.
- Quickstart: Show enabling video, disabling via --no-video, and verifying outputs.
- Agent file update: run update-agent-context.sh in /tasks step per template (documented in quickstart).

## Phase 2: Task Planning Approach
- The /tasks command will:
  - Generate tasks from contracts and data model (tests first).
  - Create micro-batch integration test task, CLI flag wiring, manifest update, renderer selection, and performance measurement tasks.
  - Order by TDD and dependencies (schema and tests → implementation → docs).

## Phase 3+: Future Implementation
Beyond /plan scope.

## Complexity Tracking
| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| — | — | — |

## Progress Tracking
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
