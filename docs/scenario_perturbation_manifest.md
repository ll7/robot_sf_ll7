# Scenario Perturbation Manifest

[← Back to Documentation Index](./README.md)

`scenario_perturbation_manifest.v1` is a small preflight surface for issue #1858 and parent
criticality issue #1610. It records the scenario, seeds, perturbation family, magnitude bounds, and
validity policy before any perturbed row is allowed into a planner pilot. Tracked manifests are
local pilot inputs unless a later command records planner execution evidence.

The schema lives at
[`robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json`](../robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json).
Tracked examples include
[`configs/scenarios/perturbations/issue_1858_seed_sensitive_pilot_v1.yaml`](../configs/scenarios/perturbations/issue_1858_seed_sensitive_pilot_v1.yaml)
and the exploratory density input
[`configs/scenarios/perturbations/issue_1610_pedestrian_density_offset_pilot_v1.yaml`](../configs/scenarios/perturbations/issue_1610_pedestrian_density_offset_pilot_v1.yaml).

## Contract

The executable v1 schema/preflight path currently supports:

- `noop`: the unperturbed baseline row for paired comparisons;
- `robot_route_offset`: a bounded `(dx_m, dy_m)` shift applied to selected robot route waypoints;
- `pedestrian_route_offset`: the same bounded waypoint shift applied to selected pedestrian routes;
- `single_pedestrian_start_delay_offset`: a bounded start-delay offset for explicit
  `single_pedestrians`;
- `single_pedestrian_speed_offset`: a bounded speed offset for explicit `single_pedestrians`;
- `single_pedestrian_wait_duration_offset`: a bounded `wait_at.wait_s` offset for explicit
  single-pedestrian wait entries;
- `pedestrian_density_offset`: a bounded `simulation_config.ped_density` offset for scenarios
  with pedestrian routes.

Issue #1959 adds a tracked exploratory pilot input for `pedestrian_density_offset`:
[`configs/scenarios/perturbations/issue_1610_pedestrian_density_offset_pilot_v1.yaml`](../configs/scenarios/perturbations/issue_1610_pedestrian_density_offset_pilot_v1.yaml).
That manifest records a bounded `simulation_config.ped_density` delta for
`classic_group_crossing_medium` and a fail-closed Francis wait probe that should be excluded
because it has no pedestrian routes. Treat the manifest and preflight result as local pilot input,
not planner execution or benchmark evidence.

Every manifest must include:

- `scenario_config`: the source scenario manifest;
- `seed_controls.baseline_seeds`: explicit replay seeds;
- `validity.max_route_offset_m`: manifest-level route-offset bound used by the executable v1
  preflight path;
- `validity.invalid_variant_evidence_policy: exclude_from_success_evidence`;
- one or more `variants`, each with `variant_id`, `scenario_id`, `family`, and `seeds`.

Density inputs additionally record `validity.max_pedestrian_density_delta`,
`validity.max_pedestrian_density`, and per-variant `density_delta`, `max_abs_density_delta`, and
`max_ped_density` parameters.

## Preflight

Run:

```bash
uv run python scripts/tools/preflight_scenario_perturbations.py \
  configs/scenarios/perturbations/issue_1858_seed_sensitive_pilot_v1.yaml \
  --output output/scenario_perturbation_preflight/issue_1858_seed_sensitive_pilot_v1.json \
  --fail-on-excluded
```

The preflight uses `scenario_cert.v1` after applying each supported perturbation. Rows with
`benchmark_evidence_status=eligible_success_evidence_candidate` may feed a later pilot matrix.
Rows marked `excluded_from_success_evidence` or `stress_only_not_success_evidence` must be reported
as limitations and must not count as successful benchmark evidence.

## Pilot Materialization

After preflight passes, materialize only eligible variants into a local scenario matrix:

```bash
uv run python scripts/tools/materialize_scenario_perturbation_pilot.py \
  configs/scenarios/perturbations/issue_1858_seed_sensitive_pilot_v1.yaml \
  --output-dir output/scenario_perturbation_pilot/issue_1610_seed_sensitive_pilot_v1 \
  --seed-limit 1
```

The generated `scenario_matrix.yaml` and route override files are local execution inputs for the
next paired planner pilot. They are not benchmark evidence by themselves, and excluded preflight
rows are omitted from the matrix. When `--output-dir` points inside the repository, it must be
under the ignored `output/` tree so materialized pilot inputs do not become durable provenance by
accident.

## Criticality Pilot

Run the first paired no-op versus route-offset pilot with:

```bash
uv run python scripts/validation/run_scenario_perturbation_criticality_pilot.py \
  configs/scenarios/perturbations/issue_1858_seed_sensitive_pilot_v1.yaml \
  --materialized-output-dir output/scenario_perturbation_pilot/issue_1904_seed_sensitive_pilot_v1 \
  --pilot-output-dir output/scenario_perturbation_pilot/issue_1904_seed_sensitive_pilot_v1/results \
  --seed-limit 1 \
  --horizon 80 \
  --workers 1 \
  --planner goal \
  --planner orca \
  --evidence-summary docs/context/evidence/issue_1904_scenario_perturbation_pilot_2026-05-31/summary.json
```

The pilot writes raw episode JSONL under `output/` and, when requested, a compact tracked summary
under `docs/context/evidence/`. Fallback, degraded, invalid, missing, and failed rows are reported
separately and excluded from completed-pair mean deltas.

## Boundary

This is not planner execution and not paper-facing evidence. It only proves that the selected
baseline and perturbed scenario variants are well-formed enough to run. A #1610 pilot still needs
paired planner execution with identical planner/config/seed controls except for `variant_id`, plus
explicit handling for missing, fallback, degraded, stress-only, and invalid rows.
