# Issue #1091 SDD Importer

Related issue: [#1091](https://github.com/ll7/robot_sf_ll7/issues/1091)

## Decision

Use the Stanford Drone Dataset as the first real-world trajectory candidate. The official project
page lists a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 license, and the dataset has
an established trajectory annotation format that can be imported without building a generic
multi-dataset abstraction first.

## Implementation Boundary

Implemented:

* `scripts/tools/import_sdd_scenarios.py` for local SDD `annotations.txt` files.
* YAML/JSON map-definition loading through `build_robot_config_from_scenario`.
* Generated map, scenario, and provenance outputs.
* Unit coverage with a tiny SDD-format fixture to prove importer output is loadable.
* Documentation for license assumptions, canonical command, normalization, and limitations.

Not implemented:

* Dataset download automation.
* Redistribution of SDD annotation snippets.
* A curated scenario generated from staged real SDD annotations. Follow-up:
  [#1126](https://github.com/ll7/robot_sf_ll7/issues/1126).
* Scene-specific calibrated homographies or obstacle maps.
* Generic multi-dataset adapter framework.

## Rationale

The issue comment chose the one-dataset-first option. Keeping SDD import narrow reduces licensing
and normalization risk while still establishing the benchmark-facing contract: local source data in,
versioned Robot SF scenario/map/provenance files out.

## Validation Plan

Use:

* `rtk uv run pytest tests/tools/test_import_sdd_scenarios.py -q`
* A local importer smoke command against a staged `annotations.txt` file.
* `load_scenarios(...)` and `build_robot_config_from_scenario(...)` on the generated scenario.
* Full PR readiness before handoff.
