# Scenarios Reference and Discovery CLI

[← Back to Documentation Index](./README.md) | [User Guide](./user-guide.md) | [Public API](./public_api.md)

This document describes the scenario catalog layout, discovery conventions,
and the import-light `robot-sf scenarios` CLI subcommands introduced in issue #8748.

## Overview

Robot SF scenarios declare the environment configuration, map asset references,
pedestrian trajectories, actor spawn rules, kinematics constraints, and observation
visibility settings for simulation episodes.

The `robot-sf scenarios` command family provides fast, read-only discovery, inspection,
and schema validation without importing optional simulation dependencies (such as
matplotlib, SciPy, PyTorch, Stable-Baselines3, or CARLA), registering sensors, creating
gym environments, or executing episodes.

## Commands

### 1. `robot-sf scenarios list`

Discovers all bundled scenarios reachable through the declared curated roots
(`single/`, `archetypes/`, top-level `configs/scenarios/*.yaml`, and `sets/`).

```bash
# Human-friendly table format
uv run robot-sf scenarios list

# Versioned JSON format
uv run robot-sf scenarios list --format json
```

#### Output Schema (`scenario_list.v1`)

```json
{
  "schema_version": "scenario_list.v1",
  "count": 139,
  "scenarios": [
    {
      "identity": "classic_crossing_low",
      "source_file": "configs/scenarios/archetypes/classic_crossing.yaml",
      "duplicate_sources": [],
      "family": "crossing",
      "status": "valid",
      "map_reference": {
        "map_file": "maps/svg_maps/classic_crossing.svg",
        "map_id": null,
        "route_overrides_file": null,
        "resolved_map_path": "maps/svg_maps/classic_crossing.svg",
        "map_exists": true,
        "external_asset_dependent": false,
        "external_map_asset_dependent": false,
        "external_route_asset_dependent": false
      },
      "actor_counts": {
        "ped_density": 0.02,
        "single_pedestrians": 0,
        "num_robots": 1,
        "groups": 0.0
      },
      "seed_policy": {
        "seeds": [1, 2, 3],
        "repeats": null,
        "count": 3
      },
      "horizon_and_dt": {
        "max_episode_steps": 500,
        "time_step": 0.1,
        "sim_time_s": 50.0
      },
      "kinematics_and_observation": {
        "kinematics": "differential_drive",
        "observation_visibility_enabled": true,
        "fov_degrees": 120.0,
        "max_range_m": 8.0
      },
      "validation_status": {
        "status": "valid",
        "valid": true,
        "reasons": [],
        "errors": [],
        "warnings": []
      }
    }
  ],
  "exclusions": [
    {
      "path": "configs/scenarios/archetype_validation_waivers.yaml",
      "reason": "archetype validation waivers, not a scenario manifest"
    },
    {
      "path": "configs/scenarios/archetypes/classic_density_tier_index.yaml",
      "reason": "classic archetype density tier index documentation, not a scenario manifest"
    }
  ]
}
```

### 2. `robot-sf scenarios describe <id-or-name>`

Inspects the declared properties of a scenario by exact identity, file stem, or direct file path.

```bash
# Describe by exact identity
uv run robot-sf scenarios describe quickstart_demo_crossing_basic

# Describe by file stem
uv run robot-sf scenarios describe quickstart_demo

# Describe by relative path
uv run robot-sf scenarios describe configs/scenarios/single/quickstart_demo.yaml

# Structured JSON output
uv run robot-sf scenarios describe quickstart_demo_crossing_basic --format json
```

#### Output Schema (`scenario_describe.v1`)

The describe payload extends the scenario summary with `requested_query` and `raw_metadata`:

- `identity`: Declared scenario identifier.
- `source_file`: Canonical relative path where the scenario is defined.
- `duplicate_sources`: Additional manifests or files declaring the same scenario identity.
- `family`: Declared scenario family or archetype category.
- `status`: Lifecycle/usability status (see [Status Values](#status-values)).
- `map_reference`: Map SVG path, map ID, route overrides, and local filesystem resolution.
- `actor_counts`: Spawn density, single pedestrian counts, robot count, and group fraction.
- `seed_policy`: Declared deterministic seeds or repeat counts.
- `horizon_and_dt`: Maximum episode steps, timestep interval, and computed simulation duration.
- `kinematics_and_observation`: Drive kinematics, sensor visibility, field-of-view, and sensor range.
- `validation_status`: Schema and asset validity result with reason codes.
- `raw_metadata`: Complete unparsed metadata dictionary from the scenario config.

### 3. `robot-sf scenarios validate <path>`

Validates a scenario YAML file or manifest with the canonical scenario schema and
metadata validators, map/route asset checks, and repository path security.

```bash
# Validate in human-friendly terminal format
uv run robot-sf scenarios validate configs/scenarios/single/quickstart_demo.yaml

# Validate in versioned JSON format
uv run robot-sf scenarios validate configs/scenarios/single/quickstart_demo.yaml --format json
```

#### Exit Codes

- `0`: Validation passed cleanly (valid scenario, all referenced map assets exist on disk).
- `2`: Validation failed (schema or metadata violation, missing map reference, malformed YAML, unsupported scenario, external asset dependency, or path traversal outside the repository root).

#### Output Schema (`scenario_validate.v1`)

```json
{
  "schema_version": "scenario_validate.v1",
  "target_path": "configs/scenarios/single/quickstart_demo.yaml",
  "resolved_path": "/path/to/repo/configs/scenarios/single/quickstart_demo.yaml",
  "valid": true,
  "status": "valid",
  "num_scenarios": 1,
  "scenarios": [...],
  "errors": [],
  "warnings": []
}
```

## Status Values

These are the scenario statuses emitted by the list, describe, and validation
payloads:

| Status | Meaning |
|---|---|
| `valid` | Schema valid, all referenced maps and assets exist on disk, and no conflicts are present. |
| `missing` | The scenario file or a referenced map/route override file is missing on disk. |
| `invalid` | Schema validation error, malformed YAML, empty file, path traversal, or another validation failure. |
| `duplicate` | The scenario identity appears with conflicting definitions in the canonical source class. |
| `unsupported` | Scenario explicitly declares `supported: false`; validation remains invalid and exits nonzero. |
| `external_asset_dependent` | A map, route override, or manifest include/search path resolves outside the repository; validation remains invalid and exits nonzero. |

Describe lookup failures use a separate error envelope with `status: "error"`;
that value is not a scenario status. External manifest includes and search paths
are rejected before loading; external map and route assets are reported with the
`external_asset_dependent` status and `EXTERNAL_ASSET_DEPENDENT` reason code.

## Reason Codes

Error and diagnostic reports carry stable reason codes:

- `OK`: Validation successful.
- `PATH_TRAVERSAL`: Path traverses outside the repository root boundary.
- `FILE_NOT_FOUND`: Referenced scenario file does not exist on disk.
- `IS_A_DIRECTORY`: Expected a scenario file path, but target is a directory.
- `EMPTY_FILE`: Scenario file is empty.
- `MALFORMED_YAML`: YAML syntax parsing failure.
- `LOAD_FAILURE`: Manifest include cycle, unresolvable includes, or invalid manifest structure.
- `SCHEMA_VALIDATION_ERROR`: Field or manifest-metadata validation failed against the [canonical scenario schema](../robot_sf/benchmark/schemas/scenarios.schema.json) or its validator.
- `DUPLICATE_SCENARIO_ID`: Duplicate scenario identifier declared within the file.
- `MAP_REFERENCE_MISSING`: Scenario declares neither `map_file` nor `map_id`.
- `MAP_NOT_FOUND`: Referenced registry `map_id` or map file does not resolve to an existing map on disk.
- `ROUTE_OVERRIDES_NOT_FOUND`: Referenced route overrides file does not exist on disk.
- `UNSUPPORTED_SCENARIO`: Scenario explicitly declared with `supported: false`.
- `EXTERNAL_ASSET_DEPENDENT`: A declared map, route override, manifest include, or map search path leaves the repository boundary.

The canonical item schema permits extension keys, but this CLI's curated
scenario contract rejects unknown top-level row fields with
`SCHEMA_VALIDATION_ERROR`; nested values are still checked by the canonical
validator.

Friendly output is a human-readable projection of the same facts exposed by JSON.
It includes a deterministic `Contract facts` section containing every payload
field, nested value, status, reason, warning, and diagnostic; JSON remains the
machine-readable contract, so field ordering and whitespace are not significant.

## Curated Roots and Exclusion Registry

The scenario catalog scans the following directories in precedence order:
1. `configs/scenarios/single/`: Canonical individual scenario definitions.
2. `configs/scenarios/archetypes/`: Canonical archetype interaction definitions.
3. `configs/scenarios/*.yaml`: Top-level curated matrices and manifests.
4. `configs/scenarios/sets/`: Curated scenario sets and evaluation slices.

Each YAML candidate is classified and expanded by the canonical scenario
loader, including includes, selection, overrides, and map-registry rebasing.
Auxiliary mappings without manifest keys are skipped; malformed manifests and
mixed mapping/non-mapping rows fail closed instead of being silently dropped.
Manifest references outside the repository, including symlink targets that resolve
outside the repository, are rejected before an external file is read. Included
manifests are passed through the canonical metadata and item-schema validators.
Mixed malformed rows retain row-level diagnostics and valid neighbors in validate
reports. A map or route asset outside the repository is explicitly classified as
`external_asset_dependent`; the map reference also records whether the map or the
route asset caused that classification.

Files explicitly excluded from the runnable scenario catalog:
- `configs/scenarios/archetype_validation_waivers.yaml`: Waiver specification registry.
- `configs/scenarios/archetypes/classic_density_tier_index.yaml`: Machine-readable documentation index.
- `configs/scenarios/README.md`: Documentation.
- `configs/scenarios/debug_scenario.bash`: Diagnostic shell wrapper.
- `configs/scenarios/contracts/`: Contract definitions.
- `configs/scenarios/maps/`: Map assets and metadata.
- `configs/scenarios/perturbations/`: Perturbation definitions.
- `configs/scenarios/results/`: Historical results cache.
- `configs/scenarios/route_overrides/`: Route override definitions.
- `configs/scenarios/templates/`: Draft scenario authoring templates.
- `configs/scenarios/generated/`: Synthetic generated scenario archives.
- `configs/scenarios/sets/vulnerable_user_proxy_pack_v0_deferred_issue3654.yaml`: Deferred synthetic proxy scaffold.

## Programmatic Python API

The CLI features are also accessible programmatically:

```python
from robot_sf.cli_scenarios import (
    list_scenarios_payload,
    describe_scenario_payload,
    validate_scenario_payload,
)

# List catalog
catalog = list_scenarios_payload()
print(f"Total scenarios: {catalog['count']}")

# Describe scenario
desc = describe_scenario_payload("quickstart_demo_crossing_basic")
print(f"Map file: {desc['map_reference']['map_file']}")

# Validate single scenario or manifest
report = validate_scenario_payload("configs/scenarios/single/quickstart_demo.yaml")
assert report["valid"] is True
```
