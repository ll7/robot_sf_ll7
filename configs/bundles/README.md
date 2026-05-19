# Task Bundles

Task bundles are named, versioned packages of existing scenario YAML files. They provide a small
unit between a single scenario and a full benchmark matrix for training curricula, smoke sweeps, and
reviewable scenario subsets.

Schema version `robot_sf.task_bundle.v1` supports:

```yaml
schema_version: robot_sf.task_bundle.v1
name: sanity-smoke-v1
description: Compact non-paper smoke bundle for validating scenario plumbing.
scenario_files:
  - ../scenarios/sanity_v1.yaml
select_scenarios:
  - planner_sanity_simple
```

- `name` is the registry key used by `bundle:<name>` references.
- `scenario_files` is a non-empty list of durable scenario YAML files. Paths resolve relative to the
  bundle file and must not point into local `output/`.
- `select_scenarios` is optional. When present, it filters the expanded bundle by scenario
  `name`, `scenario_id`, or `id` in the listed order.

Use bundles anywhere the include-aware scenario loader is used:

```bash
uv run robot_sf_bench list-scenarios --matrix bundle:sanity-smoke-v1
uv run robot_sf_bench validate-config --matrix bundle:sanity-smoke-v1
uv run robot_sf_bench preview-scenarios --matrix bundle:sanity-smoke-v1
```

Training configs may point `scenario_config` or compatible scenario-matrix fields at
`bundle:sanity-smoke-v1` when the training entry point calls `load_scenarios`. Keep bundle contents
tracked under `configs/`; generated `output/` files are not durable bundle inputs.
