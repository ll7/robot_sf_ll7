# Implementation Plan: Architectural decoupling and consistency overhaul

**Branch**: `149-architectural-coupling-and` | **Date**: 2025-10-29 | **Spec**: ../spec.md
**Input**: Feature specification from `/specs/149-architectural-coupling-and/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Reduce tight coupling across simulator, sensors, and fusion by introducing stable interfaces (Simulator Facade, Sensor, Fusion) selected via unified config. Enforce consistent error handling and config validation. Aim for zero env code changes on backend swap and frictionless sensor addition without touching simulator/fusion internals.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.13 (repo standard)
**Primary Dependencies**: Gymnasium-compatible envs, fast-pysf SocialForce subtree, NumPy, pytest, Ruff, ty (type checker)
**Storage**: N/A (in-memory simulation; files for configs and artifacts)
**Testing**: pytest (unified suite), coverage on by default
**Target Platform**: macOS/Linux dev, headless CI
**Project Type**: Single repository with library `robot_sf/` and subtree `fast-pysf/`
**Performance Goals**: Maintain ~20–25 steps/sec baseline; no regression in env creation (<1s)
**Constraints**: Backward compatibility of env factories and schemas (Constitution VII)
**Scale/Scope**: Internal architecture refactor; no new public factories in this phase

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Gates derived from Constitution:
- II Factory-Based Abstraction: Must preserve factory entry points only.
- IV Unified Configuration: Must consolidate settings into unified config; no ad-hoc kwargs.
- VII Backward Compatibility: No breaking changes to env factories or schemas.
- IX Test Coverage: Add/maintain smoke tests for public behavior.
- XII Logging: Use central logging; avoid prints in library code.

Status: PASS preliminarily (spec aligns). To be re-checked after Phase 1 artifacts.

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

**Structure Decision**: Single project; contracts and design docs live under `specs/149-architectural-coupling-and/`. Library code adjustments will remain within `robot_sf/` and `fast-pysf/` wrappers as needed in implementation phases (not part of this planning step).

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
