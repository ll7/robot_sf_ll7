# Quickstart: Consolidate Episode Schema Definitions

**Date**: 2025-09-26
**Based on**: User scenarios from spec.md

## Success Criteria Checklist

After implementing this feature, verify these scenarios work:

### Scenario 1: Schema Consolidation Complete
```bash
# Before: Multiple identical schema files
find . -name "episode.schema.v1.json"
# Output: ./robot_sf/benchmark/schemas/episode.schema.v1.json
#         ./specs/120-social-navigation-benchmark-plan/contracts/episode.schema.v1.json

# After: Only canonical schema exists
find . -name "episode.schema.v1.json"
# Output: ./robot_sf/benchmark/schemas/episode.schema.v1.json
```

### Scenario 2: Runtime Resolution Works
```python
from robot_sf.benchmark.schema_loader import load_schema

# Load schema from canonical location
schema = load_schema("episode.schema.v1.json")
print(f"Schema version: {schema['title']}")
# Output: Schema version: RobotSF Benchmark Episode (v1)
```

### Scenario 3: Git Hook Prevents Duplicates
```bash
# Create a duplicate schema file
cp robot_sf/benchmark/schemas/episode.schema.v1.json duplicate.schema.v1.json
git add duplicate.schema.v1.json

# Attempt commit - should be blocked
git commit -m "Add duplicate schema"
# Output: ERROR: Duplicate schema detected: duplicate.schema.v1.json
#         Canonical location: robot_sf/benchmark/schemas/episode.schema.v1.json
```

### Scenario 4: Schema Version Detection
```python
from robot_sf.benchmark.schema_loader import get_schema_version

version = get_schema_version("episode.schema.v1.json")
print(f"Version: {version['major']}.{version['minor']}.{version['patch']}")
# Output: Version: 1.0.0
```

### Scenario 5: Backward Compatibility Maintained
```python
# Existing code continues to work
import json
from pathlib import Path

# Old way still works during transition
schema_path = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")
with open(schema_path) as f:
    schema = json.load(f)
print("Legacy loading still works")
```

## Implementation Validation Steps

### Step 1: Verify Schema Consolidation
- [ ] Only `robot_sf/benchmark/schemas/episode.schema.v1.json` exists
- [ ] Duplicate in `specs/120-social-navigation-benchmark-plan/contracts/` removed
- [ ] All references updated to use canonical location

### Step 2: Test Runtime Resolution
- [ ] `load_schema()` function works from any package location
- [ ] Schema caching prevents repeated file I/O
- [ ] Clear error messages for missing schemas

### Step 3: Validate Git Hook
- [ ] Pre-commit hook installed and active
- [ ] Duplicate schema files blocked from commit
- [ ] Legitimate schema files allowed

### Step 4: Check Version Management
- [ ] Version parsing works for v1, v2, etc.
- [ ] Breaking change detection implemented
- [ ] Semantic versioning enforced

### Step 5: Documentation Updated
- [ ] Developer guide mentions canonical schema location
- [ ] Migration notes for schema consolidation
- [ ] API documentation for new schema loading functions

## Quick Test Script

Run this script to validate the implementation:

```python
#!/usr/bin/env python3
"""Quick validation script for schema consolidation feature."""

import sys
from pathlib import Path

def main():
    print("🔍 Validating Schema Consolidation Feature")
    print("=" * 50)

    # Check 1: Only canonical schema exists
    schema_files = list(Path(".").rglob("episode.schema.v1.json"))
    canonical_path = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")

    if len(schema_files) == 1 and schema_files[0] == canonical_path:
        print("✅ Only canonical schema file exists")
    else:
        print(f"❌ Found {len(schema_files)} schema files, expected 1")
        for f in schema_files:
            print(f"   {f}")
        return 1

    # Check 2: Schema loads correctly
    try:
        from robot_sf.benchmark.schema_loader import load_schema
        schema = load_schema("episode.schema.v1.json")
        if schema.get("title") == "RobotSF Benchmark Episode (v1)":
            print("✅ Schema loads correctly via new API")
        else:
            print(f"❌ Unexpected schema title: {schema.get('title')}")
            return 1
    except ImportError:
        print("⚠️  Schema loader not implemented yet")
    except Exception as e:
        print(f"❌ Schema loading failed: {e}")
        return 1

    # Check 3: Git hook exists
    hook_path = Path(".git/hooks/pre-commit")
    if hook_path.exists():
        content = hook_path.read_text()
        if "prevent-schema-duplicates" in content:
            print("✅ Git hook configured for duplicate prevention")
        else:
            print("⚠️  Git hook exists but may not check for duplicates")
    else:
        print("⚠️  Git hook not found (may be managed by pre-commit)")

    print("\n🎉 Schema consolidation validation complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

Save as `validate_schema_consolidation.py` and run with:
```bash
uv run python validate_schema_consolidation.py
```</content>
<parameter name="filePath">/Users/lennart/git/robot_sf_ll7/specs/136-consolidate-episode-schema/quickstart.md