# Scenario Layout

This directory supports a mix of **per-scenario**, **per-archetype**, and
**manifest** files.

## Layout

- `classic_interactions.yaml`, `francis2023.yaml`: manifest entry points for the two suites.
- `classic_interactions_francis2023.yaml`: combined manifest for both suites.
- `single/`: one scenario per file (manual fine-tuning and small edits).
- `archetypes/`: groups of related scenarios (may contain one or many entries).
- `sets/`: manifest files that include other scenario files.

## Plausibility tracking

Scenario metadata includes a `plausibility` block used to record verification
status and interaction metrics:

```yaml
metadata:
  plausibility:
    status: unverified
    verified_on: null
    verified_by: null
    method: null
    notes: null
    metrics_updated_on: null
    metrics:
      min_distance: null
      mean_distance: null
      robot_ped_within_5m_frac: null
      ped_force_mean: null
      force_q95: null
```

## Manifest (include) files

Manifest files use `includes` (or `include` / `scenario_files`) to combine
per-scenario and per-archetype YAMLs into a single scenario list.

```yaml
# configs/scenarios/sets/classic_crossing_subset.yaml
includes:
  - ../archetypes/classic_crossing.yaml
map_search_paths:
  - ../../../maps/svg_maps
```

The loader expands includes relative to the manifest file and preserves order.
If `map_file` paths in included scenarios are not resolvable, you can set
`map_search_paths` to help locate map files. The loader logs a warning with
the attempted paths and suggestion.

## Map references

Scenarios can reference maps via either `map_file` (a path) or `map_id` (a
registry key). `map_id` entries are resolved through `maps/registry.yaml` and
rebased relative to the manifest root so scenarios stay portable.

```yaml
scenarios:
  - name: crossing_demo
    map_id: classic_crossing
```

If both `map_id` and `map_file` are provided, `map_id` takes precedence (with a
warning). To override the registry path, set `ROBOT_SF_MAP_REGISTRY`. You can
regenerate the registry with `scripts/tools/generate_map_registry.py`.

## Usage

Point training/analysis configs at a manifest (or a legacy combined file):

```yaml
scenario_config: ../scenarios/sets/classic_crossing_subset.yaml
```

Example:

```bash
uv run python scripts/tools/policy_analysis_run.py \
  --training-config configs/training/benchmark_orca_classic_crossing_subset.yaml \
  --policy socnav_orca
```

## Benchmark coverage notes

- `archetypes/classic_urban_crossing.yaml` adds a corner-building urban crossing layout
  to complement open-plaza crossing maps. This increases benchmark relevance for
  real-world constrained intersection flows.
