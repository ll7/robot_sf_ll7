# Issue #944 Multi-Ped Adversarial Scenario Payload

## Goal

Issue #944 adds a pure-data bridge from `adversarial-multi-ped.v1` configs to scenario-loader-ready
manifest payloads. It is a child of issue #870 and is stacked on the issue #923 schema plus issue
#936 single-pedestrian override materializer.

## Decision

`materialize_multi_ped_scenario_payload(config, scenario_template)` returns a one-scenario manifest
payload:

- generated `single_pedestrians` entries are merged into the template by pedestrian id,
- existing template fields are preserved unless overridden by the adversarial materialization,
- `seeds` and `simulation_config.route_spawn_seed` are set to the adversarial scenario seed,
- `metadata.adversarial_multi_ped` stores the config's JSON-safe provenance payload.

The helper does not load maps, run environments, certify scenarios, or make benchmark claims. It
exists so later runners can write scenario YAML without duplicating merge logic.

## Validation

Proof command for the implementation branch:

```bash
uv run pytest tests/adversarial/test_adversarial_search.py -q
```

The RED run failed during collection because `materialize_multi_ped_scenario_payload` did not exist.
After implementation, the focused adversarial test file passed with `28 passed`.
