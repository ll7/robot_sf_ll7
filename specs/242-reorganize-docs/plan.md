# Implementation Plan: Reorganize Docs

**Branch**: `242-reorganize-docs` | **Date**: 2025-11-10 | **Spec**: `/specs/242-reorganize-docs/spec.md`
**Input**: Feature specification from `/specs/242-reorganize-docs/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Reorganize the documentation under `docs/` to improve discoverability, align with the Development Guide, and ensure all public surfaces are indexed from `docs/README.md` as required by Constitution Principle VIII (Documentation as an API Surface). The change focuses on:
- Creating or updating a central docs index with clear sections and links.
- Normalizing filenames, section headings, and cross-links across guides.
- Adding lightweight quickstart guidance for finding core docs.
- Ensuring that any new or moved docs are referenced from `docs/README.md`.

No code behavior changes are planned; this is a documentation-only reorganization with zero impact on public APIs or schemas.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Markdown + Python 3.11 (repository standard per `pyproject.toml`)  
**Primary Dependencies**: N/A for docs reorg; repository tooling uses uv, Ruff, pytest (no changes)  
**Storage**: N/A (versioned docs in repository)  
**Testing**: pytest (no new tests required), link validation deferred (manual for this PR)  
**Target Platform**: Developers reading docs locally (VS Code) and on GitHub  
**Project Type**: Single repository documentation reorganization  
**Performance Goals**: N/A (documentation only)  
**Constraints**: Must comply with Constitution Principle VIII; keep central index up-to-date; avoid breaking existing links where possible  
**Scale/Scope**: Reorganize and index existing docs under `docs/`; no new code modules

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Alignment with Constitution 1.3.1:
- Principle VIII (Documentation as an API Surface): Central docs index must reference any new or moved guides. ACTION: Update `docs/README.md` accordingly. STATUS: Planned.
- Principle VII (Backward Compatibility & Evolution Gates): No public API or schema changes. STATUS: Not applicable/No impact.
- Quality & Performance Targets: Documentation-only; no runtime performance impact. STATUS: Not applicable.
- Deliverables & Workflow: Any new docs or structural changes will be linked from `docs/README.md`; no tests or schema changes required. STATUS: Planned.

Gate Result: PASS (no contract or API changes). Re-check after Phase 1 to confirm central index updated and no broken references are introduced.

## Phase 0: Research

All NEEDS CLARIFICATION items resolved in `research.md`:
- Link validation approach deferred (manual review only, optional future CI tool).
- Placeholder contracts artifact added (no runtime endpoints).
- Minimal documentation entity model defined.

Output Artifacts:
- `research.md` (decisions, rationale, alternatives)

## Phase 1: Design & Contracts

Artifacts generated:
- `data-model.md` capturing DocPage/Section/Link conceptual model.
- `contracts/openapi.yaml` placeholder (no endpoints).
- `quickstart.md` for navigation guidance.

Post-Design Constitution Check:
- Principle VIII satisfied once index updates land (planned implementation step).
- No changes to runtime contracts (Principle VII unaffected).

## Phase 2: Implementation Plan (to execute next)

Milestones:
1. Update `docs/README.md` index (add categories: Getting Started, Core Guides, Benchmarking & Metrics, Tooling, Architecture & Refactoring, Figures & Visualization, Advanced Topics).
2. Normalize headings: ensure each top-level page has purpose sentence.
3. Cross-link environment factory usage from Quickstart to `ENVIRONMENT.md` and dev guide.
4. Scan for orphaned pages (grep for markdown filenames not referenced in README) and link or deprecate.
5. Manual link validation (open changed pages; fix broken relative paths).
6. Final Constitution re-check & artifact summary.

Risks & Mitigations:
- Risk: Broken relative links after renaming → Mitigation: incremental rename with search & manual validation.
- Risk: Over-indexing (too many categories) → Mitigation: keep to 6–7 high-level sections.

Out of Scope:
- Automated link checker integration (follow-up issue).
- Documentation site generator tooling.

Success Criteria:
- All major docs reachable within two clicks from `docs/README.md`.
- No 404s on internal links in changed files.
- Quickstart provides clear navigation path.

## Project Structure

### Documentation (this feature)

```text
specs/242-reorganize-docs/
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

```text
docs/
├── README.md            # Central docs index (to be updated)
├── dev_guide.md         # Development Guide (authoritative)
├── ENVIRONMENT.md       # Environment overview
├── SIM_VIEW.md          # Simulation UI notes
├── refactoring/         # Architecture and migration notes
├── baselines/           # Baseline algorithm docs
├── snqi-weight-tools/   # SNQI tooling docs
├── dev/                 # In-progress engineering docs
└── img|figures|video/   # Visual assets
```

**Structure Decision**: Consolidate navigation through `docs/README.md`; ensure all important guides listed above are linked from appropriate sections (quick start, architecture, benchmarks, tooling). No source code directories change.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | — | — |
