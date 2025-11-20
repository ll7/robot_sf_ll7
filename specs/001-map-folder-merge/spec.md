# Finalized Specification: Map Folder Merge

**Feature Branch**: `001-map-folder-merge`  
**Created**: 2025-11-20  
**Status**: Draft  
**Source Issue**: #18 "Maps folder exists in two locations"

## Problem Statement
Duplicate map directories (`maps/` and `robot_sf/maps/`) cause confusion and hard-coded map references increase maintenance risk.

## Objective
Unify map assets under one canonical hierarchy (SVG layouts separated from JSON metadata) and ensure environments access maps only through a registry abstraction (no literals like `"uni_campus_big"`).

## User Stories
1. (P1) Maintainer consolidates duplicate folders; audit shows zero stray files.
2. (P2) Contributor adds new SVG+JSON map in <=5 minutes following docs.
3. (P3) Environment consumer selects map by ID; loading succeeds; invalid ID yields clear error.

## Edge Cases
- Missing metadata during move -> abort removal and report.
- Duplicate map ID -> abort consolidation with guidance.
- Empty metadata directory -> fail fast with guidance.
- Direct legacy path usage -> clear error pointing to canonical location.

## Functional Requirements
- FR-001 Single canonical maps root organizing SVG and JSON predictably.
- FR-002 JSON metadata relocated under dedicated metadata subdirectory.
- FR-003 No assets remain in legacy non-canonical locations (audit = 0 orphans).
- FR-004 Environments use registry/map pool only (no hard-coded keys in classes).
- FR-005 Existing map IDs preserved (backward compatibility).
- FR-006 Unknown map ID produces validation error listing available IDs.
- FR-007 Documentation updated (layout, naming, addition workflow, validation command).
- FR-008 Automated check flags stray assets outside canonical hierarchy.

## Non-Functional Requirements
- NFR-001 Init performance change <5%.
- NFR-002 Map addition workflow time <=5 minutes.
- NFR-003 Migration process atomic and revertible.

## Key Entities
- Map Asset (SVG) – layout geometry.
- Map Metadata (JSON) – zones, dimensions, semantic info.
- Map Registry – ID to layout+metadata aggregation.
- Map Pool – configuration interface enumerating IDs.

## Assumptions
- One JSON metadata file per map ID.
- Tests rely on map IDs not file paths.
- Both SVG and JSON required for complete map definition.

## Dependencies
- Environment factory & map_pool configuration.
- Existing tests/examples referencing map IDs.

## Constraints
- Backward compatibility for IDs.
- No new asset formats.
- Minimal touch to unrelated simulation code.

## Risks & Mitigations
- Missed hard-coded reference -> pattern search & audit.
- Asset deletion mistake -> staged move with rollback instructions.
- Doc lag -> update docs in same PR before merge.
- Duplicate ID -> pre-migration ID scan.

## Success Criteria
- SC-001 Audit: zero map files outside canonical hierarchy.
- SC-002 Zero hard-coded map key literals in environment classes after migration.
- SC-003 100% existing map-related tests pass unchanged.
- SC-004 Contributor trial adds map in <=5 minutes per docs.
- SC-005 Map ID count unchanged pre vs post migration.
- SC-006 Zero confusion reports about folder choice in first month post-merge.

## Acceptance Criteria Summary
- AC-001 Canonical hierarchy present; legacy directory removed.
- AC-002 Registry loads all prior IDs.
- AC-003 Hard-coded references replaced.
- AC-004 Docs updated & published.
- AC-005 Unknown ID error lists valid IDs.
- AC-006 Audit & validation scripts pass.

## Clarifications
No clarifications required; defaults applied.

## High-Level Test Approach
1. Snapshot existing IDs.
2. Move JSON metadata.
3. Update environment references.
4. Run smoke env creation for sample IDs.
5. Run full test suite.
6. Audit paths & search for hard-coded keys.

## Glossary
- Canonical: Single authoritative source.
- Map ID: Stable identifier for selecting a map.

---
End finalized specification.
