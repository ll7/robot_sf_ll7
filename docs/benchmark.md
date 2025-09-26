# Benchmark Schema Management

## Schema Consolidation

The RobotSF benchmark system uses a consolidated schema management approach to ensure consistency and prevent duplication across the codebase.

### Canonical Schema Location

**Single source of truth**: `robot_sf/benchmark/schemas/`
- Episode schemas: `episode.schema.v1.json`
- Scenario schemas: `scenarios.schema.json`

### Runtime Schema Resolution

Use the schema loader for runtime resolution:

```python
from robot_sf.benchmark.schema_loader import load_schema, get_schema_version

# Load episode schema
schema = load_schema("episode.schema.v1.json")

# Get schema version
version = get_schema_version("episode.schema.v1.json")
print(f"Schema version: {version}")  # SchemaVersion(major=1, minor=0, patch=0)
```

### Schema Validation

Schemas are automatically validated against JSON Schema draft 2020-12:

```python
from robot_sf.benchmark.validation_utils import validate_schema_integrity

errors = validate_schema_integrity(schema_data)
if errors:
    print(f"Schema validation errors: {errors}")
```

### Version Management

Schema evolution follows semantic versioning:

```python
from robot_sf.benchmark.version_utils import detect_breaking_changes, determine_version_bump

# Detect breaking changes between schema versions
breaking_changes = detect_breaking_changes(old_schema, new_schema)

# Determine appropriate version bump
bump_type = determine_version_bump(breaking_changes)  # 'major', 'minor', or 'patch'
```

### Git Hook Prevention

Git hooks prevent duplicate schema files from being committed:

```bash
# Pre-commit hook automatically blocks duplicate schemas
git add duplicate_episode_schema.json
git commit -m "Add schema"
# ERROR: Duplicate schema detected: duplicate_episode_schema.json
#        Canonical location: robot_sf/benchmark/schemas/episode.schema.v1.json
```

### Performance Characteristics

Schema loading is optimized with caching:
- First load: <50ms (typical)
- Cached loads: <1ms
- Performance budget: <100ms hard limit

### Migration Notes

- **Before**: Multiple duplicate schema files across the repository
- **After**: Single canonical schema with runtime resolution
- **Compatibility**: All existing code continues to work unchanged
- **Prevention**: Git hooks block future duplication attempts