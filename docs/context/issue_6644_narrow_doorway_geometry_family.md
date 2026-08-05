# Issue #6644: Narrow-doorway geometry-family preflight

Issue: [ll7/robot_sf_ll7#6644](https://github.com/ll7/robot_sf_ll7/issues/6644)

This package defines a diagnostic geometry family for the authored
`francis2023_narrow_doorway` scenario. It varies two factors:

- `gap_width_m`: the free width across the doorway;
- `constriction_depth_m`: the distance travelled inside the wall-lined part of the
  passage. In this generator it is the width of the two internal wall rectangles
  along the route direction.

The generated maps are temporary or caller-owned outputs. The source map and its
release rows are not changed.

## Versioned protocol

The manifest is
`configs/benchmarks/issue_6644_narrow_doorway_geometry_family_v1.yaml`.
It records the units, baseline geometry, route waypoints, map bounds, authoritative
nominal radius, reduced-radius oracle probe, planner roster, seeds, horizon, and the
no-submission boundary.

The matrix has 15 cells:

| Factor | Levels |
| --- | --- |
| `gap_width_m` | 0.8, 1.9, 2.0, 2.1, 2.2 m |
| `constriction_depth_m` | 0.25, 1.0, 2.0 m |

The continuous nominal-envelope margin is always derived as:

```text
gap_width_m - 2 * envelope_radius_m
```

The 2.0 m baseline is therefore the zero-margin boundary for the audited 1.0 m
radius. A negative margin is an impossible geometry candidate. A zero margin is a
boundary/tangent candidate. A positive margin is only a geometrically feasible
candidate; it is not a planner or safety result.

## Oracle-first output

Run the bounded preflight with:

```bash
uv run python scripts/validation/run_issue_6644_narrow_doorway_geometry_preflight.py \
  --out-json /tmp/issue_6644_geometry_preflight.json
```

The runner creates each variant in a temporary directory, runs the existing
planner-free feasibility oracle at the nominal and reduced probe radii, and then
adds a planner record with `status: not_run`. The planner record is not an empty
result claim. It states that a separate campaign packet is required.

Each cell retains:

- generated scenario and map checksums;
- the continuous geometry margin;
- the oracle category and per-radius verdicts;
- explicit planner status, fallback, degradation, and missingness fields.

The output is diagnostic only. It does not submit Slurm work, change frozen release
artifacts, or admit evidence into a manuscript claim.

## Validation

```bash
uv run ruff check robot_sf/benchmark/narrow_doorway_geometry_family.py \
  scripts/validation/run_issue_6644_narrow_doorway_geometry_preflight.py \
  tests/benchmark/test_issue_6644_narrow_doorway_geometry_family.py
uv run ruff format --check robot_sf/benchmark/narrow_doorway_geometry_family.py \
  scripts/validation/run_issue_6644_narrow_doorway_geometry_preflight.py \
  tests/benchmark/test_issue_6644_narrow_doorway_geometry_family.py
uv run pytest -q tests/benchmark/test_issue_6644_narrow_doorway_geometry_family.py
```

Production planner execution remains blocked until a separate campaign packet
declares the exact planner command, output root, capacity check, row accounting,
fallback policy, and evidence-admission decision.
