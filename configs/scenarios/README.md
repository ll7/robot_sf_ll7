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
  - ../single/classic_crossing_low.yaml
  - ../single/classic_crossing_medium.yaml
  - ../single/classic_crossing_high.yaml
map_search_paths:
  - ../../../maps/svg_maps
```

The loader expands includes relative to the manifest file and preserves order.
If `map_file` paths in included scenarios are not resolvable, you can set
`map_search_paths` to help locate map files. The loader logs a warning with
the attempted paths and suggestion.

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
