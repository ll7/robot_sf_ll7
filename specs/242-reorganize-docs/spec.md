# Feature Specification: Reorganize Documentation Index

**Feature Branch**: `242-reorganize-docs`  
**Created**: 2025-11-10  
**Status**: Complete  
**Input**: User description: "Reorganize docs index with categorized sections and improve discoverability"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Central Docs Index with Categorized Sections (Priority: P1) 🎯 MVP

Reorganize `docs/README.md` into 8 categorized sections (Getting Started, Benchmarking & Metrics, Tooling, Architecture & Refactoring, Simulation & UI, Figures & Visualization, Performance & CI, Hardware & Environment) to enable two-click navigation to key content.

**Why this priority**: Central navigation is critical for discoverability. Without categorization, developers waste time searching for documentation.

**Independent Test**: Open `docs/README.md` and verify each top-level section exists and links resolve locally.

**Acceptance Scenarios**:

1. **Given** a developer needs setup instructions, **When** they open `docs/README.md`, **Then** they see "Getting Started" section with Development Guide and Environment links
2. **Given** a developer needs benchmark info, **When** they scan the index, **Then** they find "Benchmarking & Metrics" section with all benchmark-related guides

---

### User Story 2 - Normalize H1 Headings and Purpose Statements (Priority: P2)

Ensure all major documentation files have proper H1 headings and clear purpose statements at the top for consistency and clarity.

**Why this priority**: Consistent structure improves readability and helps developers quickly determine if a document is relevant.

**Independent Test**: Check `docs/UV_MIGRATION.md`, `docs/ENVIRONMENT.md`, and other major guides for H1 heading + purpose statement.

**Acceptance Scenarios**:

1. **Given** a file like `UV_MIGRATION.md`, **When** opened, **Then** it starts with an H1 heading and one-line purpose statement
2. **Given** all major guides, **When** reviewed, **Then** they follow consistent heading structure

---

### User Story 3 - Add Cross-Links and Back-Links (Priority: P2)

Add "See also" cross-references in related documents and back-links to central index for improved navigation.

**Why this priority**: Cross-links enable discovery of related content without returning to index repeatedly.

**Independent Test**: Check `docs/benchmark.md` and `docs/SIM_VIEW.md` for "See also" sections; verify major guides have back-links to index.

**Acceptance Scenarios**:

1. **Given** `docs/benchmark.md`, **When** scrolling to bottom, **Then** "See also" section lists related docs (SNQI tools, distribution plots)
2. **Given** major guide pages, **When** opened, **Then** back-link to central index appears at top

---

### User Story 4 - Orphan Documentation Scan (Priority: P3)

Identify documentation files not explicitly linked from central index to ensure all content is discoverable.

**Why this priority**: Prevents valuable documentation from being hidden due to missing links.

**Independent Test**: Run manual scan of `docs/*.md` and compare with index; record findings in `specs/242-reorganize-docs/orphans.md`.

**Acceptance Scenarios**:

1. **Given** all `.md` files in `docs/`, **When** compared with index, **Then** orphan list is generated with recommendations
2. **Given** orphan scan results, **When** reviewed, **Then** critical docs are linked in index or marked as intentionally unlisted

---


### Edge Cases

- **Empty sections**: If a category has no docs, do not create the section (keep only populated categories)
- **Broken links**: Verify all links resolve to existing files before merging
- **Duplicate links**: ENVIRONMENT.md was listed twice - removed duplicate under "Getting Started"
- **Backup safety**: Create backup before any modifications to enable rollback

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Central index MUST categorize all major documentation files into logical sections
- **FR-002**: All major guides MUST have H1 headings and purpose statements
- **FR-003**: Related documentation MUST include cross-reference links for discoverability
- **FR-004**: Central index MUST provide back-link targets for major guides  
- **FR-005**: System MUST identify orphaned documentation files not linked from index
- **FR-006**: All links MUST be validated to resolve correctly before merging


### Key Entities *(include if feature involves data)*

- **DocPage**: Represents a markdown documentation file (attributes: file path, H1 heading, purpose statement, links to/from)
- **Section**: A category in the central index (attributes: name, description, list of DocPage links)
- **Link**: Connection between documentation pages (attributes: source page, target page, link text, link type: cross-reference/back-link)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Developers can find any major guide within 2 clicks from `docs/README.md`
- **SC-002**: All 8 categorized sections contain correct, working links to existing documentation
- **SC-003**: 100% of major guides have H1 headings and purpose statements  
- **SC-004**: Zero broken links in central index (validated manually before merge)
- **SC-005**: Orphan scan identifies all unlisted docs with recommendations for inclusion

