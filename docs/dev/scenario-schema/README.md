# Scenario Matrix Schema

This document describes the JSON Schema used to validate benchmark scenario matrices for `robot_sf_bench`.

- Schema file: `robot_sf/benchmark/schema/scenarios.schema.json`
- Validator helper: `robot_sf/benchmark/scenario_schema.py`
- CLI validation: `robot_sf_bench validate-config --matrix <path>.yaml`
- Current manifest metadata version: `robot_sf.scenario_matrix.v1`

A scenario matrix is a YAML (or JSON) list of scenario objects. Minimal valid entry:

```yaml
- id: demo-uni-low-open
  density: low        # low | med | high
  flow: uni           # uni | bi | cross | merge
  obstacle: open      # open | bottleneck | maze
  repeats: 2          # integer >= 1
```

Optional fields:
- groups: number in [0.0, 1.0]
- speed_var: low | med | high
- goal_topology: point | swap | circulate
- robot_context: ahead | behind | embedded

Manifest-style matrices may include `schema_version: robot_sf.scenario_matrix.v1` next to
`includes`, `scenario_files`, `select_scenarios`, or `scenarios`. Unsupported versions fail early
in `validate-config` with an error at `/schema_version`.

Notes:
- The CLI also checks for duplicate `id` values.
- The validator uses Draft-07 JSON Schema.

### `simulation_config.ped_density` unit

`ped_density` is a pedestrian spawn density in **pedestrians per square meter of
spawnable route/zone (sidewalk) area** — not per whole-map area and not per route
meter. The runtime spawn count is computed as
`ceil(total_sidewalks_area * ped_density)`
(`robot_sf/ped_npc/ped_population.py`), where `total_sidewalks_area` is the
spawnable sidewalk area derived from the map routes. The classic archetypes use
`0.02` (low) / `0.05` (medium) / `0.08` (high) on this unit; the CLI flags values
outside the recommended `[0.02, 0.08]` band and flags values above `0.15` as
likely unit confusion (e.g. a value meant per whole-map area or per route meter).

The value `0.0` is a **marker-controlled placeholder**, not an empty scene. When
`metadata.spawn_mode` is `markers`, pedestrians are placed from fixed map
(`single_ped`) markers and/or config `single_pedestrians`, and `ped_density: 0.0`
simply disables route-density spawning. This is how the bottleneck archetypes
control pedestrian counts. Outside marker mode, `ped_density: 0.0` means no
route-spawned pedestrians.

The schema enforces `ped_density >= 0`; negative values fail validation.

Example:

```bash
uv run robot_sf_bench validate-config --matrix configs/baselines/example_matrix.yaml
```

This prints a JSON report with `num_scenarios`, `errors`, `warnings`, and source metadata, and
exits non-zero on errors.
