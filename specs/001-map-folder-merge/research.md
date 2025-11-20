# Phase 0 Research: Map Folder Merge

## Overview
The specification declares no clarifications required. This research phase documents explicit decisions, validates absence of unknowns, and records alternatives considered for traceability.

## Decisions

### D1: Canonical Asset Hierarchy
- **Decision**: Use top-level `maps/` as canonical root; create `maps/metadata/` for JSON files; keep `maps/svg_maps/` for SVG layouts.
- **Rationale**: Maintains discoverability (root-level assets), separates geometry (SVG) from semantics (JSON), aligns with existing docs (`SVG_MAP_EDITOR.md`).
- **Alternatives Considered**:
  - Consolidate under `robot_sf/maps/` (rejected: buries assets; harder for contributors to find).
  - Mixed directory per map ID containing both SVG+JSON (rejected: duplicates path scanning logic; less scalable for future asset types).

### D2: Registry Abstraction Placement
- **Decision**: Implement `robot_sf/maps/registry.py` loading all maps by ID.
- **Rationale**: Keeps code-level logic inside package per Principle XI (library reuse). Avoids environment modules performing filesystem traversal.
- **Alternatives**:
  - Inline logic in `environment_factory.py` (rejected: bloats factory; harder to test in isolation).
  - Global singleton object (rejected: complicates test determinism; explicit function calls preferred).

### D3: Backward Compatibility Strategy
- **Decision**: Preserve all existing map IDs; introduce validation helper `validate_map_id(id)` raising informative error enumerating valid IDs.
- **Rationale**: Satisfies Principle VII; prevents silent failure; improves contributor UX.
- **Alternatives**:
  - Soft warning fallback (rejected: may hide typos and degrade reproducibility).

### D4: Migration Workflow
- **Decision**: Two-step atomic move: (1) copy assets to new hierarchy, (2) run audit to confirm zero stray files, (3) remove legacy `robot_sf/maps/` asset copies.
- **Rationale**: Enables rollback before deletion; ensures SC-001 (zero outside hierarchy).
- **Alternatives**:
  - In-place rename (rejected: higher risk of partial moves; less auditable).

### D5: Audit Tooling
- **Decision**: Implement `robot_sf/maps/audit.py` (or extend existing common helpers) to scan for stray SVG/JSON outside canonical paths and search for hard-coded map ID literals in `robot_sf/gym_env/`.
- **Rationale**: Automates SC-001 and SC-002; integrates with tests.
- **Alternatives**:
  - Manual grep documented in README (rejected: non-deterministic; prone to omissions).

### D6: Performance Considerations
- **Decision**: Cache registry build (list + metadata parse) once at first access; subsequent environment creations reuse cached dict.
- **Rationale**: Ensures NFR-001 (<5% init delta) by avoiding repeated filesystem traversals.
- **Alternatives**:
  - Rebuild per environment creation (rejected: unnecessary overhead).

### D7: Testing Strategy
- **Decision**: Add tests: `test_maps_registry.py` for: (a) list IDs matches pre-migration snapshot, (b) invalid ID raises expected error, (c) audit returns zero stray assets.
- **Rationale**: Satisfies Principle IX; guards regressions.
- **Alternatives**:
  - Rely solely on existing environment tests (rejected: insufficient specificity for asset layout changes).

### D8: Documentation Updates
- **Decision**: Update `docs/SVG_MAP_EDITOR.md` and `docs/README.md` index; add quickstart instructions in `specs/001-map-folder-merge/quickstart.md`.
- **Rationale**: Principle VIII; ensures contributors can add maps within target time.
- **Alternatives**:
  - Only update dev guide (rejected: map addition is contributor task; needs direct doc entry).

## Unresolved Items
None (no NEEDS CLARIFICATION markers remained).

## Risks Revisited
| Risk | Mitigation Confirmed | Residual Risk |
|------|----------------------|---------------|
| Missed hard-coded reference | Automated grep + audit test | Low |
| Asset deletion mistake | Two-phase move + pre-deletion snapshot | Low |
| Doc lag | Docs updated in same PR gate | Low |
| Duplicate ID | Pre-migration scan of metadata filenames/IDs | Low |

## Next Steps
Proceed to Phase 1: create `data-model.md`, contracts, quickstart, then implement migration & tests.
