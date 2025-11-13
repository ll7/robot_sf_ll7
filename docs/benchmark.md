# Benchmark Schema Management

[← Back to Documentation Index](./README.md)

## Schema Consolidation

The RobotSF benchmark system uses a consolidated schema management approach to ensure consistency and prevent duplication across the codebase.

> **Looking for runnable examples?** See `examples/benchmarks/demo_full_classic_benchmark.py`
> for a programmatic walkthrough and consult [`examples/README.md`](../examples/README.md)
> for the full benchmarks catalog.

**See also**: [SNQI Weight Tools](./snqi-weight-tools/README.md) for weight recomputation and optimization, and [Distribution Plots](./distribution_plots.md) for visualization guidance.

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

## Algorithm Grouping & Aggregation Diagnostics

The classic benchmark aggregates metrics **per algorithm**. To guarantee separation:

- Episode writers (classic orchestrator, CLI resume path, smoke scripts) **must** populate both the top-level `algo` field and the nested mirror `scenario_params["algo"]`. The orchestrator now enforces this contract and raises `robot_sf.benchmark.AggregationMetadataError` when the metadata is missing or malformed.
- Aggregation utilities (`compute_aggregates`, `compute_aggregates_with_ci`) prefer the nested key, fall back to the top-level `algo`, and fail fast if both are absent. When an expected baseline never appears, aggregation continues but
  - emits a Loguru warning with `event="aggregation_missing_algorithms"`, and
  - annotates the JSON summary with `_meta.missing_algorithms`, `_meta.group_by`, and `_meta.effective_group_key` (`"scenario_params.algo | algo | scenario_id"`).
- Episode injection logs expose observability hooks:
  - `event="episode_metadata_injection"` (nested value added) and
  - `event="episode_metadata_mismatch"` (nested value corrected to match top-level `algo`).

### Validation Checklist

- Spot-check the first line of `episodes.jsonl`: `record["algo"] == record["scenario_params"]["algo"]`.
- Confirm aggregate outputs include `_meta.effective_group_key` and, when applicable, warnings describing any missing algorithms.
- Treat `AggregationMetadataError` as a signal to regenerate the episode data—legacy files lacking mirrored metadata are no longer accepted silently.
