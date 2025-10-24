# Scenario Matrix Schema

This document describes the JSON Schema used to validate benchmark scenario matrices for `robot_sf_bench`.

- Schema file: `robot_sf/benchmark/schema/scenarios.schema.json`
- Validator helper: `robot_sf/benchmark/scenario_schema.py`
- CLI validation: `robot_sf_bench validate-config --matrix <path>.yaml`

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

Notes:
- The CLI also checks for duplicate `id` values.
- The validator uses Draft-07 JSON Schema.

Example:

```bash
uv run robot_sf_bench validate-config --matrix configs/baselines/example_matrix.yaml
```

This prints a JSON report with `num_scenarios`, `errors`, and `warnings`, and exits non‑zero on errors.