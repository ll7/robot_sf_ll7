# Issue #870 Multi-Ped Adversarial Runtime Slice

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/870>

Predecessor notes:

- [Issue #923 Multi-Ped Adversarial Candidate Schema](issue_923_multi_ped_adversarial_schema.md)
- [Issue #936 Multi-Ped Adversarial Overrides](issue_936_multi_ped_adversarial_overrides.md)
- [Issue #944 Multi-Ped Adversarial Scenario Payload](issue_944_multi_ped_adversarial_scenario_payload.md)

## Goal

This slice turns the existing `adversarial-multi-ped.v1` pure-data contract into a first runtime
path that can build a `RobotSimulationConfig`, reset the Robot-SF environment, and step with N>1
scripted adversarial pedestrians.

## Public Surface

- `robot_sf.adversarial.runtime.multi_ped_config_to_single_pedestrian_definitions(config)`
- `robot_sf.adversarial.runtime.validate_multi_ped_runtime_plausibility(config, base_map, ...)`
- `robot_sf.adversarial.runtime.build_multi_ped_adversarial_robot_config(config, base_map, ...)`
- The same helpers are re-exported from `robot_sf.adversarial`.

The runtime builder deep-copies the supplied base `MapDefinition`, replaces its
`single_pedestrians` with the generated adversarial pedestrians, sets zero background pedestrian
density, fixes the route-spawn seed to the adversarial scenario seed, and selects the generated map
via `RobotSimulationConfig.map_id`.

## Runtime Guardrails

The new runtime check fails closed before environment construction when:

- the schema-level config is invalid,
- a scripted pedestrian speed exceeds the conservative runtime cap,
- start or goal coordinates are outside the base map bounds,
- start or goal coordinates are inside an obstacle or below the configured obstacle clearance,
- two scripted pedestrians start closer than the configured start-separation threshold.

Per-pedestrian adversarial metadata is now carried through `SinglePedestrianDefinition.metadata`,
scenario-loader `single_pedestrians` overrides, and `populate_single_pedestrians(...)` metadata.

## Scope Boundary

This started as a runtime smoke path, not the complete #870 epic. Follow-up slices have now added
two scripted development families and a policy-analysis smoke proof, but generated scenarios are
still development stress tests unless a separate certification path promotes them.

Still out of scope:

- learned adversary training,
- replay certification,
- benchmark-frozen case promotion.

Treat generated scenarios as development stress tests until those later proof surfaces exist.

## Validation

TDD evidence for this branch:

```bash
uv run pytest tests/adversarial/test_adversarial_search.py -q
```

The RED run failed during collection with `ModuleNotFoundError: No module named
'robot_sf.adversarial.runtime'`. After implementation, the same focused file passed with
`32 passed`; the reset/step runtime smoke was the slowest case at about 19 seconds.

Before PR handoff, run the broader targeted and readiness checks:

```bash
uv run pytest tests/adversarial/test_adversarial_search.py tests/test_multi_pedestrian.py -q
uv run ruff check robot_sf/adversarial/runtime.py robot_sf/adversarial/__init__.py \
  robot_sf/nav/map_config.py robot_sf/ped_npc/ped_population.py \
  robot_sf/training/scenario_loader.py tests/adversarial/test_adversarial_search.py
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

2026-05-06 update: a focused policy-analysis smoke now materializes a temporary multi-ped
`group_squeeze` scenario against the existing `static_humans` map hooks (`h1`, `h2`), runs
`scripts/tools/policy_analysis_run.py` through its Python `main(...)` entry point with the `goal`
policy, and asserts that:

- the produced episode is not an error-record fallback,
- the episode has real rollout steps,
- `scenario_params.metadata.adversarial_multi_ped_runtime` survives the policy-analysis record,
- per-pedestrian attribution metadata survives the materialized `single_pedestrians` block.

Validation command:

```bash
uv run pytest \
  tests/tools/test_policy_analysis_run.py::test_policy_analysis_main_runs_materialized_multi_ped_adversarial_scenario -q
```

Observed result:

```text
1 passed in 40.39s
```

The runtime test coverage also now includes an explicit N=1 adversarial reset/step smoke for the
same 1-N contract:

```bash
uv run pytest \
  tests/adversarial/test_adversarial_search.py::test_multi_ped_adversarial_runtime_config_resets_and_steps_single_pedestrian \
  tests/adversarial/test_adversarial_search.py::test_multi_ped_adversarial_runtime_config_resets_and_steps -q
```

Observed result:

```text
2 passed in 41.82s
```
