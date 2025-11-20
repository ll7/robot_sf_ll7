# Phase 1 Data Model: Map Folder Merge

## Entities Overview

### 1. MapAsset
Represents the SVG layout file for a map.
- Fields:
  - `id: str` (unique, stable)
  - `svg_path: Path` (canonical path under `maps/svg_maps/`)
  - `checksum: str` (optional; future integrity verification)
- Relationships:
  - 1:1 with `MapMetadata` (via `id`)
- Validation:
  - File exists and is readable
  - ID derived from filename (without extension) matches registry record

### 2. MapMetadata
Semantic and dimensional metadata for a map.
- Fields:
  - `id: str` (matches `MapAsset.id`)
  - `json_path: Path` (canonical path under `maps/metadata/`)
  - `dimensions: tuple[float, float]` (width, height; optional if embedded in JSON)
  - `zones: list[dict[str, Any]]` (semantic regions)
  - `version: str` (schema or asset revision tag)
- Validation:
  - JSON parse succeeds
  - Required keys present: at minimum `id`
  - `id` matches filename base and asset ID

### 3. MapRegistry
Aggregates all maps by ID, providing lookup and listing functions.
- Fields:
  - `maps: dict[str, MapRecord]` where `MapRecord = {"asset": MapAsset, "metadata": MapMetadata}`
  - `built_at: float` (timestamp)  
  - `source_paths: list[Path]` (used for audit)
- Operations:
  - `build_registry(root: Path) -> MapRegistry`
  - `get(id: str) -> MapRecord`
  - `list_ids() -> list[str]`
  - `validate_map_id(id: str) -> None` (raises ValueError with available IDs)
- Validation:
  - Every `MapAsset` has matching `MapMetadata`
  - No duplicate IDs

### 4. MapPool (Config Layer)
Existing config interface enumerating map IDs for scenario selection.
- Fields (existing):
  - `map_pool: list[str]`
- Integration Change:
  - Ensure selected IDs validated through `MapRegistry.validate_map_id` at environment creation.

## State Transitions

| Transition | Trigger | Result | Validation |
|------------|---------|--------|-----------|
| BUILD_REGISTRY | First access or cache miss | Registry populated | All IDs unique; asset/metadata pairs complete |
| ADD_MAP (future extension) | Developer adds new SVG+JSON | Registry rebuild required | New ID not already present |
| REMOVE_MAP (migration) | During consolidation | Legacy path removed | New canonical pair exists with same ID |

## Derived / Computed Fields
- `checksum` (optional future) can be computed during registry build for integrity.
- `built_at` recorded to allow performance monitoring and caching decisions.

## Validation Rules Summary
1. No duplicate IDs.
2. Each SVG has corresponding JSON with matching ID.
3. Invalid ID lookup raises descriptive error listing valid IDs.
4. Audit ensures zero map files outside canonical directories.

## Open Questions
None (captured in research; no NEEDS CLARIFICATION items remain).

## Implementation Notes
- Initial implementation excludes `checksum` to minimize scope; placeholder field documented for future integrity feature without breaking contract.
- Registry build should be isolated for unit testing (pass a temporary root with fixture assets).
