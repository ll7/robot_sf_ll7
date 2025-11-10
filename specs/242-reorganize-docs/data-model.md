# Data Model: Documentation Reorganization

Purpose: Define a minimal vocabulary for reorganizing docs so that structure and acceptance criteria are clear.

## Entities

- DocPage
  - id: string (path under `docs/`, e.g., `dev_guide.md`)
  - title: string (first H1 or inferred)
  - categories: list[string] (e.g., "guides", "benchmarks", "tooling")
  - links_out: list[Link]
  - referenced_by_index: bool (reachable from `docs/README.md`)

- Section
  - page_id: DocPage.id
  - heading: string
  - anchors: list[string]

- Link
  - source_page: DocPage.id
  - target: string (relative path or anchor)
  - type: enum {internal, external}

## Validation Rules

- Every DocPage in scope MUST be reachable from `docs/README.md` (referenced_by_index == true).
- Internal links SHOULD use relative paths and valid anchors.
- Top-level guides MUST start with an H1 and contain a short purpose statement near the top.

## Scope Notes

- No database/filesystem schema changes—this is a conceptual model guiding reorganization.
- Not enforced by code in this feature; used for review checklist.
