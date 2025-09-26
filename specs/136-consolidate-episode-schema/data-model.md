# Data Model: Consolidate Episode Schema Definitions

**Date**: 2025-09-26
**Entities Extracted From**: spec.md functional requirements FR-001 through FR-008

## Core Entities

### EpisodeSchema
**Purpose**: JSON schema definition that validates episode metrics data structure
**Fields**:
- `$schema`: JSON Schema draft version (string, required)
- `$id`: Unique schema identifier URI (string, required)
- `title`: Human-readable schema name (string, required)
- `type`: Root type, always "object" (string, required)
- `additionalProperties`: Whether extra properties allowed (boolean, required)
- `required`: List of mandatory property names (array of strings, required)
- `properties`: Schema for each property (object, required)

**Validation Rules**:
- Must be valid JSON Schema draft 2020-12
- Must include version const field
- Must define episode_id, scenario_id, seed, metrics as required
- Properties must be properly typed (string, integer, number, object, array)

**State Transitions**: Schema evolution through semantic versioning (major.minor.patch)

### SchemaReference
**Purpose**: Runtime mechanism for loading schema from canonical location
**Fields**:
- `schema_path`: Relative path from package root (string, required)
- `version`: Schema version identifier (string, required)
- `loaded_schema`: Cached schema content after first load (object, optional)

**Validation Rules**:
- Path must exist relative to robot_sf package
- Version must follow semantic versioning format
- Schema must be valid JSON when loaded

**State Transitions**:
- Unloaded → Loading → Loaded (with caching)
- Error states: FileNotFound, InvalidJSON, SchemaInvalid

### SchemaVersion
**Purpose**: Version identifier for schema evolution and compatibility tracking
**Fields**:
- `major`: Major version number (integer, required, >= 0)
- `minor`: Minor version number (integer, required, >= 0)
- `patch`: Patch version number (integer, required, >= 0)
- `prerelease`: Prerelease identifier (string, optional)
- `build`: Build metadata (string, optional)

**Validation Rules**:
- Follows semantic versioning specification (semver.org)
- Major version changes indicate breaking changes
- Minor version changes indicate backward-compatible additions
- Patch version changes indicate backward-compatible fixes

**State Transitions**:
- Version bump triggered by schema content analysis
- Breaking change detection compares structure, not just content

## Entity Relationships

```
EpisodeSchema ────contains───► SchemaVersion
    │                           (embedded in filename/title)
    │
    └──is loaded by───► SchemaReference
                        │
                        └──caches───► EpisodeSchema (runtime)
```

## Data Flow

1. **Schema Definition**: EpisodeSchema defined in canonical JSON file
2. **Version Identification**: SchemaVersion embedded in filename and content
3. **Runtime Loading**: SchemaReference loads EpisodeSchema on demand
4. **Validation**: Loaded schema used to validate episode data structures
5. **Evolution**: Schema changes trigger version updates and compatibility checks

## Validation Constraints

### EpisodeSchema Constraints
- Must be parseable as valid JSON
- Must conform to JSON Schema specification
- Required fields must be present and correctly typed
- Optional fields must have proper defaults or be nullable

### SchemaReference Constraints
- Canonical path must be resolvable at runtime
- Schema file must exist and be readable
- Loaded schema must pass JSON Schema validation
- Version must be extractable from schema content

### SchemaVersion Constraints
- Must follow X.Y.Z format where X, Y, Z are non-negative integers
- Major version 0 indicates unstable API
- Version changes must reflect actual compatibility impact</content>
<parameter name="filePath">/Users/lennart/git/robot_sf_ll7/specs/136-consolidate-episode-schema/data-model.md