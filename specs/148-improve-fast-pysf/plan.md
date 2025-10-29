# Implementation Plan: Improve fast-pysf Integration

**Branch**: `148-improve-fast-pysf` | **Date**: 2025-10-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/148-improve-fast-pysf/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Integrate fast-pysf physics tests into the main pytest suite, resolve all 24 PR #236 review comments, extend quality tooling (ruff, ty, coverage) to fast-pysf/, and improve type annotations while maintaining functional correctness. This ensures the fast-pysf subtree meets the same quality standards as robot_sf core code and enables unified testing, linting, and type checking workflows.

## Technical Context

**Language/Version**: Python 3.13.1 (project standard, already in use)  
**Primary Dependencies**: 
- pytest 8.3.4 (test framework)
- ruff 0.9.2+ (linting/formatting)
- ty (type checking via uvx)
- uv (package manager and task runner)
- fast-pysf subtree (NumPy 1.26.4, numba 0.60.0 for JIT-compiled physics)

**Storage**: N/A (test fixtures use file-based map JSON in fast-pysf/tests/test_maps/)  
**Testing**: pytest with coverage plugin, headless mode via environment variables (DISPLAY=, MPLBACKEND=Agg, SDL_VIDEODRIVER=dummy)  
**Target Platform**: macOS (dev), Linux (CI), headless execution required  
**Project Type**: Single Python library with integrated subtree (fast-pysf/ as git subtree)  
**Performance Goals**: 
- Test suite execution < 5 minutes total (robot_sf + fast-pysf combined)
- Fast-pysf tests < 60 seconds
- No degradation to existing ~43 robot_sf tests performance

**Constraints**: 
- Must maintain backward compatibility with FastPysfWrapper integration point
- Type annotations cannot break numba JIT compilation (@njit decorated functions)
- Ruff and ty configurations must not conflict with existing robot_sf settings
- All tests must run headless for CI compatibility

**Scale/Scope**: 
- Current: ~43 tests in tests/, 12 tests in fast-pysf/tests/ (2 failing due to missing fixtures)
- Target: 55+ unified tests, all passing
- PR #236 review comments: 24 issues (7 high, 10 medium, 7 low priority)
- Estimated effort: 18-25 hours across 4 phases

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

[Gates determined based on constitution file]

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```
# [REMOVE IF UNUSED] Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# [REMOVE IF UNUSED] Option 2: Web application (when "frontend" + "backend" detected)
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

# [REMOVE IF UNUSED] Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure: feature modules, UI flows, platform tests]
```

**Structure Decision**: [Document the selected structure and reference the real
directories captured above]

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
