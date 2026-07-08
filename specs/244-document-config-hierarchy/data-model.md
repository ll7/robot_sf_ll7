# Data Model: Configuration Hierarchy

**Feature**: 244-document-config-hierarchy  
**Date**: 2025-11-11  
**Status**: Complete

## Overview

This feature is primarily documentation-focused with minimal code changes. The "data model" consists of the existing configuration class hierarchy and the relationships between configuration sources.

## Entities

### 1. Configuration Source (Conceptual Entity)

Represents the three levels where configuration values originate.

**Fields**:
- `level`: str - One of "code_default", "yaml_file", "runtime_parameter"
- `precedence`: int - Priority level (1=lowest, 3=highest)
- `mutability`: str - "immutable" (code defaults), "version_controlled" (YAML), "ephemeral" (runtime)

**Relationships**:
- A Configuration Source feeds into a Config Class instance
- Higher precedence sources override lower precedence

**Validation Rules**:
- `level` must be one of the three defined levels
- `precedence` matches level mapping: code_default=1, yaml_file=2, runtime_parameter=3

### 2. Config Class (Code Entity)

Represents Python dataclasses that define configuration structure.

**Fields** (metadata about the class):
- `module_path`: str - Full import path (e.g., "robot_sf.gym_env.unified_config")
- `class_name`: str - Class identifier (e.g., "RobotSimulationConfig")
- `status`: str - "canonical" or "legacy"
- `base_class`: str | None - Parent class name if hierarchical
- `fields`: list[ConfigField] - Dataclass fields with types and defaults

**Example Instances**:

**Canonical Classes**:
- `BaseSimulationConfig` (module: unified_config, status: canonical, base: None)
- `RobotSimulationConfig` (module: unified_config, status: canonical, base: BaseSimulationConfig)
- `ImageRobotConfig` (module: unified_config, status: canonical, base: RobotSimulationConfig)
- `PedestrianSimulationConfig` (module: unified_config, status: canonical, base: RobotSimulationConfig)

**Legacy Classes**:
- `BaseEnvSettings` (module: env_config, status: legacy, base: None)
- `RobotEnvSettings` (module: env_config, status: legacy, base: BaseEnvSettings)
- `EnvSettings` (module: env_config, status: legacy, base: RobotEnvSettings) [Note: may be alias/duplicate]
- `PedEnvSettings` (module: env_config, status: legacy, base: RobotEnvSettings)

**Relationships**:
- Legacy classes map 1:1 to canonical equivalents for migration
- All classes extend from a base (either BaseSimulationConfig or BaseEnvSettings)
- Config classes are instantiated by factory functions

**Validation Rules**:
- `status` must be "canonical" or "legacy"
- Legacy classes must have a documented canonical replacement
- All classes must define default values for required fields

### 3. ConfigField (Component of Config Class)

Represents individual configuration parameters within a class.

**Fields**:
- `field_name`: str - Parameter identifier (e.g., "sim_config", "robot_config")
- `field_type`: str - Python type annotation (e.g., "SimulationSettings", "int | None")
- `default_value`: Any - Default from `field(default=...)` or `field(default_factory=...)`
- `required`: bool - Whether field must be set (no default)
- `description`: str - Docstring or inline comment explaining purpose

**Example**:
```python
# From RobotSimulationConfig
ConfigField(
    field_name="peds_have_obstacle_forces",
    field_type="bool",
    default_value=False,
    required=False,
    description="Enable pedestrian-robot physics interaction forces"
)
```

**Validation Rules**:
- `field_name` must be valid Python identifier
- `default_value` type must match `field_type`
- `required=True` implies `default_value=None` or raises at instantiation

### 4. MigrationMapping (Documentation Entity)

Represents the conversion path from legacy to canonical config classes.

**Fields**:
- `legacy_class`: str - Deprecated class name
- `canonical_class`: str - Recommended replacement
- `field_mappings`: dict[str, str] - Legacy field → canonical field name map
- `behavioral_changes`: list[str] - Any differences in runtime behavior
- `code_example`: str - Working migration snippet

**Example**:
```python
MigrationMapping(
    legacy_class="EnvSettings",
    canonical_class="RobotSimulationConfig",
    field_mappings={
        "sim_config": "sim_config",       # Direct match
        "lidar_config": "lidar_config",   # Direct match
        "robot_config": "robot_config",   # Direct match
        "map_pool": "map_pool",           # Direct match
        "render_scaling": "render_scaling"  # Direct match
    },
    behavioral_changes=[],  # No known differences
    code_example="""
# Before (legacy)
from robot_sf.gym_env.env_config import EnvSettings
config = EnvSettings()

# After (canonical)
from robot_sf.gym_env.unified_config import RobotSimulationConfig
config = RobotSimulationConfig()
"""
)
```

**Validation Rules**:
- All fields in `legacy_class` must appear in `field_mappings` or be documented as removed
- `canonical_class` must exist in unified_config module
- `code_example` must be syntactically valid Python

### 5. DeprecationWarning (Code Entity)

Represents the warning emitted when legacy classes are instantiated.

**Fields**:
- `warning_category`: str - Always "DeprecationWarning"
- `message_template`: str - Text including replacement class name
- `stack_level`: int - Call stack depth to show in traceback (always 2)
- `trigger_location`: str - Method where warning is emitted (always "__post_init__")

**Example**:
```python
DeprecationWarning(
    warning_category="DeprecationWarning",
    message_template="{legacy_class} is deprecated and will be removed in a future version. Use {canonical_class} from robot_sf.gym_env.unified_config instead.",
    stack_level=2,
    trigger_location="__post_init__"
)
```

**Validation Rules**:
- `message_template` must include both legacy and canonical class names
- `stack_level=2` ensures warning points to user code, not config class
- Warning must not prevent instantiation (non-breaking)

## State Transitions

### Config Class Lifecycle

```
[Code Default] → [YAML Override] → [Runtime Override] → [Active Config Instance]
```

**States**:
1. **Code Default**: Defined in dataclass field defaults
2. **YAML Override**: Loaded from scenario/baseline YAML file (optional)
3. **Runtime Override**: Passed as kwargs to factory function (optional)
4. **Active Config Instance**: Final merged configuration used by environment

**Transitions**:
- **Code Default → YAML Override**: When YAML file specifies a field value
- **YAML Override → Runtime Override**: When factory kwargs specify a field value
- **Any State → Active Config Instance**: After all overrides applied, instance is created

### Legacy Config Deprecation Lifecycle

```
[In Use] → [Deprecated with Warning] → [Migration Period] → [Removed]
```

**States**:
1. **In Use**: Legacy class used without warnings (pre-244)
2. **Deprecated with Warning**: Legacy class emits DeprecationWarning on instantiation (244 implementation)
3. **Migration Period**: Users migrate to canonical classes (ongoing)
4. **Removed**: Legacy class deleted from codebase (future, Phase 3)

**Transitions**:
- **In Use → Deprecated**: Feature 244 adds warnings to `__post_init__`
- **Deprecated → Migration Period**: Documentation published, users begin migration
- **Migration Period → Removed**: After multiple release cycles, legacy code deleted

**Invariants**:
- Legacy classes remain functional during deprecation (non-breaking)
- Canonical classes exist before legacy deprecation begins
- Migration guide published before removal announced

## Relationships Diagram

```
┌─────────────────────────────────────┐
│   Configuration Sources             │
│  (Conceptual - precedence order)    │
├─────────────────────────────────────┤
│ 1. Code Defaults (dataclass fields) │
│ 2. YAML Files (configs/scenarios/)  │
│ 3. Runtime Params (factory kwargs)  │
└─────────────────────────────────────┘
            │
            ↓ (feeds into)
┌─────────────────────────────────────┐
│      Config Class Hierarchy          │
├─────────────────────────────────────┤
│ Canonical (unified_config.py):      │
│  - BaseSimulationConfig             │
│  - RobotSimulationConfig            │
│  - ImageRobotConfig                 │
│  - PedestrianSimulationConfig       │
│                                     │
│ Legacy (env_config.py):             │
│  - BaseEnvSettings                  │
│  - RobotEnvSettings                 │
│  - EnvSettings                      │
│  - PedEnvSettings                   │
└─────────────────────────────────────┘
            │
            ↓ (instantiated by)
┌─────────────────────────────────────┐
│    Environment Factory Functions     │
├─────────────────────────────────────┤
│  - make_robot_env()                 │
│  - make_image_robot_env()           │
│  - make_pedestrian_env()            │
└─────────────────────────────────────┘
            │
            ↓ (guided by)
┌─────────────────────────────────────┐
│   Migration Documentation           │
├─────────────────────────────────────┤
│  - docs/architecture/configuration.md│
│  - Migration examples               │
│  - Deprecation warnings             │
└─────────────────────────────────────┘
```

## Field-Level Examples

### BaseSimulationConfig Fields
- `sim_config`: SimulationSettings (required, has factory default)
- `map_pool`: MapDefinitionPool (required, has factory default)
- `lidar_config`: LidarScannerSettings (required, has factory default)
- `render_scaling`: int | None (optional, default None)
- `backend`: str (optional, default "fast-pysf")
- `sensors`: list[dict] (optional, default empty list)

### RobotSimulationConfig Additional Fields
- `robot_config`: DifferentialDriveSettings | BicycleDriveSettings (required, factory default)
- `use_image_obs`: bool (optional, default False)
- `peds_have_obstacle_forces`: bool (optional, default False)

### ImageRobotConfig Additional Fields
- `image_config`: ImageSensorSettings (required, factory default)
- `use_image_obs`: bool (overridden to True by default)

## Summary

The data model for this feature is primarily conceptual, describing the relationships between:
1. **Configuration sources** (code, YAML, runtime) with precedence rules
2. **Config classes** (canonical vs legacy) with field mappings
3. **Migration paths** from legacy to canonical
4. **Deprecation warnings** as transition mechanism

No new database entities or persistent storage models are introduced. The "data" is the configuration hierarchy itself, documented and made discoverable through the new `docs/architecture/configuration.md` file.
