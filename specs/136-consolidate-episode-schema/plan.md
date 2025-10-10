
# Implementation Plan: Consolidate Episode Schema Definitions

**Branch**: `136-consolidate-episode-schema` | **Date**: 2025-09-26 | **Spec**: specs/136-consolidate-episode-schema/spec.md
**Input**: Feature specification from `/specs/136-consolidate-episode-schema/spec.md`

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
Establish single source of truth for episode schema definitions to eliminate maintainability risks from duplicate JSON schema files. Implement runtime resolution, semantic versioning, and git hooks to prevent future duplication while maintaining backward compatibility.

## Technical Context
**Language/Version**: Python 3.x
**Primary Dependencies**: JSON Schema validation, git hooks, pathlib
**Storage**: Filesystem (JSON schema files)
**Testing**: pytest with contract tests
**Target Platform**: Cross-platform (any OS with Python)
**Project Type**: Single project
**Performance Goals**: Fast schema loading with caching
**Constraints**: Maintain backward compatibility, prevent schema duplication
**Scale/Scope**: Small scope (consolidate existing files, add resolution layer)

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution Principles Alignment:
- **Principle III (Benchmark & Metrics First)**: ✅ This feature strengthens benchmark schema consistency and standardization
- **Principle VII (Backward Compatibility)**: ✅ Design maintains stable contracts and provides migration path
- **Principle VIII (Documentation as API Surface)**: ✅ Will add clear documentation for canonical schema location and resolution API
- **Principle IX (Test Coverage)**: ✅ Includes contract tests and validation scenarios
- **Principle XII (Logging)**: ✅ No logging requirements for this schema-focused feature

No constitution violations identified. Feature aligns with core principles of reproducible research and stable contracts.

## Project Structure

### Documentation (this feature)
```
specs/136-consolidate-episode-schema/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
│   ├── schema-loader-api.v1.json
│   └── git-hook-api.v1.json
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

**Structure Decision**: Option 1 (Single project) - Python codebase with standard layout
