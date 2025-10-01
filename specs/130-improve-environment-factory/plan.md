
# Implementation Plan: Improve Environment Factory Ergonomics

**Branch**: `130-improve-environment-factory`  
**Date**: 2025-09-23  
**Spec**: `specs/130-improve-environment-factory/spec.md`  
**Tasks**: `specs/130-improve-environment-factory/tasks.md`  
**Context**: Plan updated post partial implementation + analysis pass (legacy shim removed, seed param missing, perf threshold mismatch). This document restores concrete content and realigns execution with Constitution v1.2.0.

> NOTE: Template was re-copied; this revision reconstructs plan from analysis findings.

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

## Clarifications (Session 1 – 2025-09-23)
| Item | Question | Decision | Rationale | Impact |
|------|----------|----------|-----------|--------|
| Legacy kwargs policy (FR-005/006) | Strict error vs permissive mapping? | Reinstate compatibility shim with mapping + WARNING; strict errors only when `ROBOT_SF_FACTORY_STRICT=1`. | Constitution VII (backward compatibility). | Re-wire factories to call `_factory_compat.apply_legacy_kwargs`. |
| Permissive toggle name | Which env var? | `ROBOT_SF_FACTORY_LEGACY=1` enables legacy acceptance; `ROBOT_SF_FACTORY_STRICT=1` forces error on unknown unmapped keys. | Explicit, mirrors existing env style. | Add to spec + migration guide. |
| Validation severity (FR-004) | Warning vs exception? | Use WARNING + auto-adjust (e.g., auto-enable debug when recording) except type errors. | Ergonomics over friction. | Ensure tests assert WARN + corrected state. |
| Seed support (FR-008) | What gets seeded? | Add `seed: Optional[int]`; seed Python `random`, NumPy, env RNG; store `env.unwrapped.seed_applied`. | Determinism (Principle IV). | Add implementation + test. |
| `max_episode_steps` mention | Implement now? | DEFER (remove from FR scope). | Prevent scope creep. | Mark deferred in spec. |
| Pedestrian precedence divergence | Accept explicit opt-out? | Yes, explicit `RecordingOptions(record=False)` overrides boolean convenience. | User intent clarity. | Document & test already present. |
| Performance threshold (FR-017 +5%) | Current test uses +10%. | Tighten back to +5% mean. | Align spec + protect perf. | Adjust constant & docs. |
| Option dataclasses expansion | Add ped-specific variants? | Not now (YAGNI). | Avoid fragmentation. | Revisit when >2 ped-only flags. |
| Ped factory signature order | Freeze now? | Yes; current order locked by snapshot tests. | Stability for users. | Doc in spec & migration. |

All previous NEEDS CLARIFICATION markers are resolved; spec must be updated to remove markers and reflect above decisions.

## Summary
Improve ergonomics of environment creation by:
- Explicit factory signatures (no `**kwargs`).
- Structured option objects (`RenderOptions`, `RecordingOptions`).
- Legacy compatibility shim (to be reinstated) with warnings + env var toggles.
- Deterministic seeding (`seed` param) – missing currently.
- Normalized precedence with intentional pedestrian divergence for explicit opt-out.
- Performance guard (+5% mean baseline) – test currently at +10% (to fix).
- Comprehensive docstrings + migration guide + quick reference table.

Current divergence vs spec: legacy shim inactive, seed param absent, performance threshold mismatch, incomplete docstrings. This plan enumerates remediation tasks.

## Technical Context
**Language/Version**: Python 3.13  
**Primary Dependencies**: Gymnasium-like env interface, Loguru, NumPy, Pygame (render), `fast-pysf` SocialForce submodule.  
**Storage**: N/A (in-memory; optional video outputs).  
**Testing**: Pytest (signature, normalization, performance, pedestrian precedence, future seed determinism).  
**Target Platform**: macOS + Linux (CI headless).  
**Project Type**: Research framework (single codebase).  
**Performance Goals**: Env creation mean <= baseline_mean * 1.05; cold <1s, warm <25ms typical.  
**Constraints**: Honor Principles II, IV, VII, XII; no silent breaking changes.  
**Scale/Scope**: Limited to four factories (`make_robot_env`, `make_image_robot_env`, `make_pedestrian_env`, `make_multi_robot_env`).

## Constitution Check
| Principle | Status | Notes / Action |
|-----------|--------|----------------|
| II (Factory Abstraction) | PARTIAL | Explicit sigs done; restore legacy shim. |
| IV (Unified Config & Seeds) | FAIL | Add `seed` param + propagation. |
| VII (Backward Compatibility) | FAIL | Legacy kwargs mapping removed – reinstate. |
| XII (Logging) | PASS | Uses Loguru; add test to block stray prints. |
| IX (Test Coverage) | PARTIAL | Need seed determinism + edge case tests. |
| Performance (FR-017) | PARTIAL | Threshold mismatch (10% vs 5%). |

Gate: Cannot finalize until FAIL items resolved; PARTIAL items scheduled in remediation tasks.

## Project Structure

### Documentation (Feature Directory)
```
specs/130-improve-environment-factory/
├── spec.md        (needs marker cleanup & decisions merge)
├── plan.md        (this file)
├── tasks.md       (existing; will append remediation tasks)
├── research.md    (retrofit Session 1 decisions)
├── data-model.md  (option dataclasses captured)
├── quickstart.md  (update with seed & legacy flags)
├── contracts/     (N/A for this feature – keep empty or remove if policy allows)
├── perf_diff.md   (T024 to add once guard tightened)
└── coverage_checklist.md (T025)
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

## Phase 0: Outline & Research (Retroactive Update)
Actions:
1. Append Session 1 decisions table to `research.md`.
2. Add performance measurement methodology (sample size=30, discard first run, compute mean+stdev+p95).
3. Document pedestrian precedence divergence rationale.

Exit Criteria: research.md updated; no remaining NEEDS CLARIFICATION markers.

## Phase 1: Design & Contracts (Status: Partial)
Additions required:
- Design contract for `seed` parameter (inputs, seeding sequence, stored attribute, error modes).
- Sequence diagram: user kwargs → legacy shim → normalization → env creation.
- Validation mapping table (condition → auto-adjust → log level) per FR-004.
- Update quickstart with examples (legacy vs strict mode; seeding usage).

## Phase 2: Task Planning (Done – augment instead of regenerate)
Will append remediation tasks (T029+). Avoid re-indexing existing tasks to preserve history.

## Phase 3+: Implementation & Remediation Roadmap
| New ID | Title | Depends | Severity | Description |
|--------|-------|---------|----------|-------------|
| T029 | Reinstate legacy shim integration | T011 | CRITICAL | Re-wire factories to call `apply_legacy_kwargs`; add tests for mapped + unknown keys (warn vs strict). |
| T030 | Introduce `seed` parameter + deterministic test | T016 | CRITICAL | Add `seed` to all factories; propagate to RNGs; test identical first obs when same seed. |
| T031 | Tighten performance guard to 5% | T015 | HIGH | Adjust test constant; update baseline notes & add perf_diff.md. |
| T032 | Extend docstrings & perf notes | T014 | HIGH | Full parameter docs + precedence explanation + performance note. |
| T033 | Edge case tests (headless+debug, missing video_path) | T010 | MEDIUM | Assert warning and graceful defaults. |
| T034 | Coverage checklist | T029 | MEDIUM | Map each FR → code/tests; flag deferred FRs. |
| T035 | Migration guide & decision log | T018 | HIGH | Create `migration.md` (before/after, env vars, precedence). |
| T036 | Repr/readability & side-effect smoke | T006 | LOW | Ensure factory import has no side-effects; simple repr checks. |
| T037 | Logging no-print enforcement test | T014 | MEDIUM | Fail test if `print(` appears in factory module (except guarded). |
| T038 | Spec update removing deferred `max_episode_steps` | T001 | LOW | Mark FR deferred; adjust spec & coverage. |

Completion Criteria:
- Milestone M1 (T029, T030): All FAIL gates resolved.
- Milestone M2 (T031, T032, T035): High severity tasks done; docs updated.
- Milestone M3 (T033, T034, T036–T038): Medium/Low complete; ready for final validation.

## Complexity Tracking
Temporary dual-interface complexity (explicit signature + legacy shim) justified by Principle VII. Simpler alternative (removal) rejected: would break existing user code without deprecation window.


## Progress Tracking
**Phase Status**:
- [ ] Phase 0: Research (retrofit decisions)
- [ ] Phase 1: Design (seed + validation table)
- [x] Phase 2: Task planning (baseline tasks present)
- [x] Phase 3: Partial impl (T001–T017 complete)
- [ ] Phase 3 Remediation (T029–T038)
- [ ] Phase 4: Validation (post-remediation full suite)
- [ ] Phase 5: Finalization (coverage + migration docs)

**Gate Status**:
- [ ] Constitution Gate (Initial) – pending legacy + seed
- [ ] Constitution Gate (Post-Design) – re-eval after M1
- [x] Clarifications resolved (spec updates pending)
- [x] Complexity deviation documented

**Milestones**:
- M1: Legacy + seed (CRITICAL)
- M2: Perf guard tightened + docstrings + migration
- M3: Coverage & polish complete

---
*Based on Constitution v1.2.0 – see `.specify/memory/constitution.md`*
