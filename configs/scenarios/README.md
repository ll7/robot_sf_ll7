# Scenario Layout

This directory supports a mix of **per-scenario**, **per-archetype**, and
**manifest** files.

## Layout

- `classic_interactions.yaml`, `francis2023.yaml`: manifest entry points for the two suites.
- `classic_interactions_francis2023.yaml`: combined manifest for both suites.
- `single/`: one scenario per file (manual fine-tuning and small edits).
- `archetypes/`: groups of related scenarios (may contain one or many entries).
- `sets/`: manifest files that include other scenario files.

## Manifest (include) files

Manifest files use `includes` (or `include` / `scenario_files`) to combine
per-scenario and per-archetype YAMLs into a single scenario list.

```yaml
# configs/scenarios/sets/classic_crossing_subset.yaml
includes:
  - ../single/classic_crossing_low.yaml
  - ../single/classic_crossing_medium.yaml
  - ../single/classic_crossing_high.yaml
```

The loader expands includes relative to the manifest file and preserves order.

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
