# Issue #1083 Sanity V1 Nominal Matrix

Issue: [#1083](https://github.com/ll7/robot_sf_ll7/issues/1083)

Status date: 2026-05-09

## Goal

Add a versioned nominal-scenario matrix that complements the stress-oriented paper and h500
benchmark surfaces. `sanity_v1` is a calibration matrix: it is meant to show whether baseline
planners can handle low-ambiguity, deployment-like motion before harder stress failures are
interpreted as planner incapability.

## Added Surface

- Scenario matrix: `configs/scenarios/sanity_v1.yaml`
- Smoke campaign: `configs/benchmarks/sanity_v1_smoke.yaml`
- Kinematics: `differential_drive`
- Seed budget: one fixed seed, `111`
- Core smoke planners: `goal`, `orca`

The matrix includes four scenarios:

- `planner_sanity_simple`
- `empty_map_8_directions_east`
- `goal_behind_robot`
- `single_ped_crossing_orthogonal`

## Design Constraints

The matrix is intentionally constrained:

- at least four scenarios,
- low density or sparse interaction,
- existing parser-validated map assets only,
- no new map geometry,
- no paper-facing replacement claims,
- no stress-suite interpretation.

## Nominal Threshold

For the first smoke profile, the nominal target is:

- all configured planner rows complete without runner failure,
- no collision rows in the smoke campaign,
- aggregate core success should be visibly stronger than stress-matrix expectations.

This threshold is deliberately modest because `sanity_v1` is a calibration surface, not a promotion
gate. The smoke config uses `goal` and `orca` as the representative baseline-safe pair. A candidate
scan showed the current `social_force` row fails most verified-simple scenes at both 250 and 500
steps, so including it would make the nominal surface measure a SocialForce limitation instead of
the issue's easier deployment-like scenario intent. If future work wants a stricter multi-baseline
gate, it should first clarify whether SocialForce is expected to pass this surface.

## Validation Commands

Validate and preview the scenario matrix:

```bash
uv run robot_sf_bench validate-config --matrix configs/scenarios/sanity_v1.yaml
uv run robot_sf_bench preview-scenarios --matrix configs/scenarios/sanity_v1.yaml
```

Run the smoke campaign:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/sanity_v1_smoke.yaml \
  --mode run \
  --output-root output/benchmarks/issue_1083 \
  --campaign-id sanity_v1_smoke
```

Observed local proof on 2026-05-09:

- `validate-config`: 4 scenarios, no errors; expected warn-only notes for zero pedestrian density
  and missing density metadata.
- `preview-scenarios`: 4 scenarios, no errors; same warn-only notes.
- `sanity_v1_smoke`: 2 planner rows, 8 total episodes, 2 successful runs, `benchmark_success=true`.
- Planner table: `goal` success `1.0000`, collisions `0.0000`; `orca` success `1.0000`,
  collisions `0.0000`.

Generated reports stayed under `output/benchmarks/issue_1083/` and are reproducible from the
tracked config, seed, command, and commit. They are not durable dependencies for later runs.

## Boundaries

`sanity_v1` does not replace `paper_experiment_matrix_v1` or the h500 scenario-horizon matrix. It
should be cited only as nominal calibration evidence. Hard-matrix failures remain meaningful only
when interpreted alongside scenario certification, route-clearance warnings, fallback/degraded
planner status, and the relevant stress-surface evidence.
