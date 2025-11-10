# Research: Reorganize Docs

This document consolidates decisions and rationale to resolve all NEEDS CLARIFICATION items for the "Reorganize Docs" feature.

## Decision 1: Central docs index update (docs/README.md)
- Decision: Update `docs/README.md` to serve as the single entry point with a clear, categorized index of all important guides (Development Guide, Environment overview, Benchmarking, SNQI tools, Refactoring notes, Figures).
- Rationale: Constitution Principle VIII requires a discoverable entry for any public surface; `.github/copilot-instructions.md` also states the central point to link new docs is `docs/README.md`.
- Alternatives considered:
  - Keep current scattered links: Rejected due to discoverability problems and Constitution VIII.
  - Introduce a doc site generator (MkDocs/Sphinx): Out of scope for this patch; adds tooling and CI complexity.

## Decision 2: Link validation approach
- Decision: Prefer manual verification for this change; optionally add a lightweight CI link checker in a follow-up (tracked separately).
- Rationale: Scope is reorganization without new external links; manual review suffices to keep links intact.
- Alternatives considered:
  - Immediate integration of a link checker: Useful, but adds CI churn; defer to a follow-up PR.

## Decision 3: Contracts folder for this feature
- Decision: Provide a placeholder OpenAPI file indicating that no service endpoints are introduced by this feature.
- Rationale: The planning workflow expects `/contracts/` artifacts. This feature is documentation-only; the placeholder clarifies that no API changes are part of this work.
- Alternatives considered:
  - Omit contracts entirely: Could confuse reviewers/scripts expecting the directory; keeping a placeholder is clearer.

## Decision 4: Data model (documentation entities)
- Decision: Model documentation entities minimally as DocPage, Section, and Link to guide consistent structure.
- Rationale: Although not code, a shared vocabulary clarifies the reorganization work and acceptance criteria (e.g., every Section must be reachable from the central index).
- Alternatives considered:
  - No data model: Acceptable, but the minimal model helps review and consistency.

## Decision 5: Quickstart guidance for docs users
- Decision: Provide a small `quickstart.md` explaining where to start and how to navigate docs locally and on GitHub.
- Rationale: Reduces time-to-first-doc for contributors per Dev Guide.
- Alternatives considered:
  - Rely on README only: Quickstart adds explicit, copyable steps and expectations.

## Open Questions (tracked)
- Link validation in CI: Defer to a separate change. If adopted, prefer a fast, cacheable tool.
- Any additional doc categories to elevate in the index (e.g., GPU setup): Will be decided during index update pass.
