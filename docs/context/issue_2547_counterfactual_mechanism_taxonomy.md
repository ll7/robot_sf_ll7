# Issue #2547 Counterfactual Mechanism Taxonomy

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2547>

## Scope

Issue #2547 extends the controlled counterfactual scenario-pair manifest so each generated pair
includes an explicit mechanism taxonomy row and a why-first-friendly pair report. The implementation
updates `scripts/tools/create_counterfactual_scenario_pair.py` and keeps the existing
preflight-validated `counterfactual_scenario_pair.v1` boundary.

This is smoke-level taxonomy/reportability infrastructure. It does not infer causality from one
pair and does not promote benchmark-strength mechanism evidence.

## Contract

The manifest now includes:

- `changed_factor`, mirroring the existing `changed_feature` field for report consumers;
- `mechanism_taxonomy.label`, validated against `counterfactual_mechanism_taxonomy.v1`;
- `mechanism_taxonomy.mechanism_hypothesis`;
- `mechanism_taxonomy.expected_metric_direction`;
- `mechanism_taxonomy.validity_constraints`;
- `pair_report.base_scenario_id`;
- `pair_report.counterfactual_scenario_id`;
- `pair_report.artifact_manifest_ref`;
- `pair_report.expected_vs_observed_metric_change`.

Supported features currently include:

- `robot_route_offset`, mapped to the `clearance_pressure` mechanism label.
- `occluder_timing_offset` (Issue #3369), mapped to the `occlusion_exposure` mechanism label after
  preflight proves the source scenario is an occluded-emergence fixture with static occlusion,
  fixture timing metadata, and an explicit `emerging_ped` target.

Because the pair generator does not execute a smoke run, the expected-vs-observed section is
deliberately `not_available` with the reason
`no smoke-run metrics were supplied to this pair manifest`.

## Evidence

Tracked compact evidence:

- [evidence/issue_2547_counterfactual_mechanism_taxonomy/summary.json](evidence/issue_2547_counterfactual_mechanism_taxonomy/summary.json)

The targeted fixture in `tests/tools/test_create_counterfactual_scenario_pair.py` verifies the
taxonomy row, expected metric directions, validity constraints, pair report, and fail-closed
expected-vs-observed status.

## Validation

```bash
rtk uv run pytest tests/tools/test_create_counterfactual_scenario_pair.py -q
rtk uv run ruff check scripts/tools/create_counterfactual_scenario_pair.py tests/tools/test_create_counterfactual_scenario_pair.py
```

Issue #3369 additionally validates the occluder-timing manifest and preflight/materialization
surface through `tests/scenario_certification/test_scenario_perturbation_preflight.py`.

Expected result: targeted generator tests pass and Ruff reports no issues.

## Claim Boundary

The output can feed why-first reports as mechanism-hypothesis input. It remains candidate input only:
future causal or benchmark claims require smoke-run or benchmark evidence with repeated seeds,
baselines, denominators, and fail-closed fallback/degraded row handling.
