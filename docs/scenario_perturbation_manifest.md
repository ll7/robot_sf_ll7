# Scenario Perturbation Manifest

[← Back to Documentation Index](./README.md)

`scenario_perturbation_manifest.v1` is a small preflight surface for issue #1858 and parent
criticality issue #1610. It records the scenario, seeds, perturbation family, magnitude bounds, and
validity policy before any perturbed row is allowed into a planner pilot. Tracked manifests are
local pilot inputs unless a later command records planner execution evidence.

The schema lives at
[`robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json`](../robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json).
Tracked examples include
[`configs/scenarios/perturbations/issue_1858_seed_sensitive_pilot_v1.yaml`](../configs/scenarios/perturbations/issue_1858_seed_sensitive_pilot_v1.yaml),
the single-pedestrian trajectory input
[`configs/scenarios/perturbations/issue_1610_single_ped_trajectory_waypoint_offset_pilot_v1.yaml`](../configs/scenarios/perturbations/issue_1610_single_ped_trajectory_waypoint_offset_pilot_v1.yaml),
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
- `single_pedestrian_trajectory_waypoint_offset`: a bounded `(dx_m, dy_m)` shift applied to all
  coordinate trajectory waypoints for one explicit `single_pedestrians` entry;
- `pedestrian_density_offset`: a bounded `simulation_config.ped_density` offset for scenarios
  with pedestrian routes.

`single_pedestrian_trajectory_waypoint_offset` is intentionally narrow: it requires
`parameters.pedestrian_id`, `parameters.waypoint_selector: all`, and an existing non-empty
coordinate `trajectory`. Route pedestrians, role-only pedestrians, goal-only pedestrians, POI-only
trajectory materialization, and individual waypoint indexes fail closed until separately scoped.

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

Family-specific validity caps are required when a manifest uses the matching family:

- `validity.max_start_delay_offset_s` for `single_pedestrian_start_delay_offset`;
- `validity.max_wait_duration_offset_s` for `single_pedestrian_wait_duration_offset`;
- `validity.max_single_pedestrian_speed_delta_m_s` for `single_pedestrian_speed_offset`;
- `validity.max_single_pedestrian_trajectory_waypoint_offset_m` for
  `single_pedestrian_trajectory_waypoint_offset`.
- `validity.max_pedestrian_density_delta` for `pedestrian_density_offset`.

Density inputs also record optional `validity.max_pedestrian_density` plus per-variant
`density_delta`, `max_abs_density_delta`, and optional `max_ped_density` parameters.

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

## Perturbation Family Registry

The reusable family registry at
[`robot_sf/scenario_certification/perturbation_family_registry.py`](../robot_sf/scenario_certification/perturbation_family_registry.py)
defines each family's semantic boundary, target surface, validity constraints, fail-closed rules,
and required/optional parameter keys. Downstream writers (preflight, criticality_summary.v1)
should use the registry instead of hardcoding family knowledge.

```python
from robot_sf.scenario_certification import (
    perturbation_families,
    perturbation_family,
    supported_perturbation_families,
    validate_perturbation_family_parameters,
)

family = perturbation_family("robot_route_offset")
assert family.target_surface == "robot_route_waypoints"
reasons, family = validate_perturbation_family_parameters(
    "robot_route_offset",
    {"dx_m": 0.25, "dy_m": 0.0, "max_magnitude_m": 0.5},
)
```

## Criticality Summary v1

The criticality summary schema at
[`robot_sf/benchmark/schemas/criticality_summary.v1.json`](../robot_sf/benchmark/schemas/criticality_summary.v1.json)
and writer at
[`robot_sf/scenario_certification/criticality_summary.py`](../robot_sf/scenario_certification/criticality_summary.py)
provide a validated surface for #1610 perturbation-family summary payloads.
Every summary must record explicit row status counts for `completed`, `invalid`, `fallback`,
`degraded`, `missing`, and `failed` rows. Non-completed rows are tracked separately and never
contribute to completed-pair effect means.

Build a summary from pilot records:

```python
from robot_sf.scenario_certification import (
    build_criticality_summary_from_pilot,
    criticality_summary_to_dict,
    validate_criticality_summary,
)

summary = build_criticality_summary_from_pilot(
    records_by_planner=...,
    scenario_metadata=...,
    manifest="configs/scenarios/perturbations/example.yaml",
    manifest_id="example_v1",
    planners=["goal", "orca"],
    horizon=80,
    dt=0.1,
    seed_limit=5,
    materialization=...,
    planner_runs=...,
)
payload = criticality_summary_to_dict(summary)
validate_criticality_summary(payload)
```

Or wrap an existing compact #1610 evidence payload (without raw output paths):

```python
from robot_sf.scenario_certification import (
    build_criticality_summary_from_compact_evidence,
)

evidence = json.loads(path.read_text())
summary = build_criticality_summary_from_compact_evidence(evidence)
payload = criticality_summary_to_dict(summary)
validate_criticality_summary(payload)
```

## Boundary

This is not planner execution and not paper-facing evidence. It only proves that the selected
baseline and perturbed scenario variants are well-formed enough to run. A #1610 pilot still needs
paired planner execution with identical planner/config/seed controls except for `variant_id`, plus
explicit handling for missing, fallback, degraded, stress-only, and invalid rows.
