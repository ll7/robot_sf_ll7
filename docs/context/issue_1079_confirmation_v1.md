# Issue 1079 Confirmation v1 Scenario Matrix

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1079>

## Scope

`configs/scenarios/confirmation_v1.yaml` defines a compact confirmation matrix for ranking-stability
checks outside the current Francis-heavy paper matrix. It is intentionally additive and does not
change the paper matrix, planner defaults, or benchmark runner semantics.

## Scenario Selection Rationale

The matrix selects eight scenarios from the issue-596 atomic and sparse-interaction packs:

| Scenario | Primary capability | Why included |
| --- | --- | --- |
| `empty_map_8_directions_east` | frame consistency | Open-map baseline for coordinate and heading sanity. |
| `corner_90_turn` | topology | Simple right-angle route topology without pedestrian confounds. |
| `u_trap_local_minimum` | topology | Local-minimum escape case that is semantically distinct from crossing scenes. |
| `single_obstacle_circle` | static avoidance | Central obstacle detour with smooth-clearance expectation. |
| `narrow_passage` | static avoidance | Tight-gap centering check with explicit small robot radius. |
| `symmetry_ambiguous_choice` | robustness | Symmetric bypass case for side-choice stability. |
| `single_ped_crossing_orthogonal` | sparse dynamic interaction | Minimal single-pedestrian crossing, not a full Francis scenario package. |
| `head_on_interaction` | sparse dynamic interaction | Sparse head-on negotiation with one pedestrian. |

This covers frame, topology, static-clearance, robustness, and sparse interaction archetypes while
avoiding new real-world replay imports or a larger protocol orchestration change.

## Canonical Commands

Validate and preview the scenario matrix:

```bash
uv run robot_sf_bench validate-config --matrix configs/scenarios/confirmation_v1.yaml
uv run robot_sf_bench preview-scenarios --matrix configs/scenarios/confirmation_v1.yaml
```

Run a small core-planner smoke configuration:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/confirmation_v1_smoke.yaml \
  --mode preflight \
  --output-root output/benchmarks/issue_1079 \
  --campaign-id confirmation_v1_smoke_preflight
```

## Interpretation Boundary

`confirmation_v1` is a confirmation and robustness-check matrix, not a replacement paper matrix.
Ranking changes should be reported conservatively as evidence about robustness across this distinct
scenario subset, not as automatic superiority across the full benchmark suite.

## Validation Notes

Validated on 2026-05-09:

- `uv run pytest tests/benchmark/test_confirmation_v1_matrix.py -q` passed (`2 passed`).
- `uv run pytest tests/benchmark/test_confirmation_v1_matrix.py tests/test_scenario_schema.py
  tests/contract/test_scenario_matrix_schema.py tests/benchmark/test_runner_scenario_matrix_manifest.py
  tests/benchmark/test_camera_ready_campaign.py -q` passed (`67 passed`).
- `uv run robot_sf_bench validate-config --matrix configs/scenarios/confirmation_v1.yaml`
  returned `num_scenarios: 8` with no errors.
- `uv run robot_sf_bench preview-scenarios --matrix configs/scenarios/confirmation_v1.yaml`
  returned `num_scenarios: 8` under the warn-only preview policy.
- `uv run python scripts/tools/run_camera_ready_benchmark.py --config
  configs/benchmarks/confirmation_v1_smoke.yaml --mode preflight --output-root
  output/benchmarks/issue_1079 --campaign-id confirmation_v1_smoke_preflight`
  generated the expected ignored preflight artifacts:
  `validate_config.json`, `preview_scenarios.json`, `matrix_summary.json`,
  `matrix_summary.csv`, AMV coverage, and comparability reports.

The validation and preview warnings are the expected atomic-fixture warnings for `ped_density=0.0`,
recommended-density range, and missing `metadata.density`; they are not schema errors and match the
static/no-pedestrian intent of several issue-596 atomic scenarios.
