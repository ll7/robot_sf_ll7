# Feature Specification: Bulk SVG Map Validation & Loader Tests

**Feature Branch**: `131-add-bulk-svg`  
**Created**: 2025-09-23  
**Status**: Draft  
**Input**: User description: "Add bulk SVG map validation & loader tests and documentation"

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a developer maintaining robot-sf map assets, I want an automated (manual-trigger) script-based validation that loads either a single SVG or all SVG maps in the `maps/svg_maps/` directory so I can quickly detect missing required elements (robot routes, pedestrian routes, spawn/goal zones) and optionally auto‑augment or flag inconsistencies before using them in demos or benchmarks.

### Acceptance Scenarios
1. **Given** a valid SVG map containing at least one `robot_route_<spawn>_<goal>` path and required spawn/goal zones, **When** the manual validation script (`examples/svg_map_example.py`) is executed, **Then** it reports successful conversion with counts of routes/zones and no errors.
2. **Given** an SVG map missing all robot routes, **When** the validation script runs, **Then** it logs a validation error and (for lenient mode) continues or (for strict mode) raises an exception, clearly indicating the deficiency.
3. **Given** a directory containing a mix of valid and invalid SVG maps and lenient loading, **When** bulk load is invoked, **Then** only valid maps appear in the returned dictionary and invalid ones are logged as skipped.
4. **Given** the same directory and strict mode, **When** bulk load is invoked, **Then** the operation fails fast with an error referencing the first invalid file encountered (or aggregated failure) so the user can fix assets.
5. **Given** an SVG defining routes whose spawn/goal indices exceed available zones, **When** loading occurs, **Then** synthetic placeholder zones are generated (as per current fallback logic) and a warning is emitted.
6. **Given** an SVG with pedestrian route requirements (ped routes + spawn/goal zones), **When** validation runs, **Then** their counts are reported similarly to robot elements.

### Edge Cases
- Directory path exists but contains zero SVG files → strict mode must raise an error; lenient mode should raise as well (no usable maps). 
- Non-SVG file passed to single-file loader → error raised early with clear message.
- Duplicate filename stems in directory (e.g., copying maps) → the later one overwrites the earlier; a warning SHOULD be issued (proposed) to highlight replacement.
- Extremely minimal valid map (one robot spawn, one robot goal, one simple route) → still accepted.
- Corrupted SVG (XML parse failure) → logged as an error; skipped in lenient mode.

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: Provide a manual validation/inspection script capable of loading a single SVG map and printing success/failure status (extension of existing `svg_map_example.py`).
- **FR-002**: Enable bulk loading of all SVG maps in `maps/svg_maps/` returning a dict keyed by filename stem.
- **FR-003**: Loader MUST validate presence of at least one robot route; absence constitutes a failure (exception in strict mode, skip otherwise).
- **FR-004**: Loader MUST log counts of robot routes, ped routes, spawn zones, goal zones for each successfully converted map.
- **FR-005**: In lenient mode, loader MUST skip invalid maps and continue processing remaining files.
- **FR-006**: In strict mode, loader MUST raise on the first invalid map (propagating validation error details).
- **FR-007**: When route indices exceed available spawn/goal zones, system MUST create synthetic fallback zones and log a warning (existing behavior retained).
- **FR-008**: Manual script MUST optionally (flag or code branch) iterate all maps, summarizing validity outcomes.
- **FR-009**: Documentation MUST describe required SVG labeling schema (already present) and reference new bulk validation usage.
- **FR-010**: Colour codes in example maps SHOULD align with guidance, but functional validation is based on labels, not colours.
- **FR-011**: Provide ability to toggle strict vs lenient mode in the manual script (e.g., via env var or constant near top of file).
- **FR-012**: If zero valid maps are loaded from a directory in lenient mode, the loader MUST raise an error to prevent silent empty operation.
- **FR-013**: The validation summary MUST clearly list invalid maps with reasons (missing routes, parse error, etc.).
- **FR-014**: The manual script MUST exit with non-zero code when strict validation fails (to enable future CI hook if desired).

### Key Entities
- **SVG Map**: An authored vector asset containing labeled rectangles (zones/obstacles) and paths (routes) abiding by naming convention.
- **MapDefinition**: In-memory representation (robot & ped zones, routes, obstacles) produced by conversion.
- **Bulk Load Result**: Mapping[str filename_stem → MapDefinition].

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details beyond minimal script expectations
- [x] Focused on user value (asset correctness & validation confidence)
- [x] Written accessibly
- [x] Mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain (none present)
- [x] Requirements are testable and unambiguous
- [x] Success criteria measurable (counts, presence/absence, error conditions)
- [x] Scope bounded (bulk loading + manual script + docs update)
- [x] Dependencies/assumptions identified (existing parser & logging)

## Execution Status
- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (none outstanding)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist final pass pending

---
