# CLI Reference (Installed Entry Points)

Plain-language summary: this is the complete list of installed commands,
generated from `pyproject.toml` and live `--help` so the docs cannot drift
from the packaged entry points. See the [Glossary](glossary.md) for
acronyms and project terms.

> Generated file — do not edit by hand. Regenerate with
> `uv run python scripts/dev/generate_cli_reference.py`.
> Sources: `pyproject.toml [project.scripts]` plus
> `docs/cli_reference_meta.yaml` plus live `--help` (local-only, no network,
> simulator, scheduler, or artifact mutation).

## Overview

| Command | Purpose | Profile | Availability | Guide | Help |
| --- | --- | --- | --- | --- | --- |
| `robot-sf` | Top-level user workflow entry point (doctor, demo, examples, gallery, models, datasets, envs, scenarios, planners, recipe, release). | core | Always available after `uv sync --all-extras` (core dependencies only); `--help` is local-only with no network, simulator, scheduler, or artifact mutation. | [adoption_path.md](adoption_path.md) | available |
| `robot-sf-carla-docker-runtime` | Preflight or smoke-test the pinned CARLA 0.9.16 Docker runtime. | carla | Preflight is host-local; `smoke` and `live-replay` need the Docker daemon, the pinned `carlasim/carla:0.9.16` image, and the `carla` client group; `--help` is local-only. | [dev_runtime_requirements.md](dev_runtime_requirements.md) | available |
| `robot-sf-carla-parity-bundle-preflight` | Read-only readiness preflight for the compact CARLA native/aligned parity bundle. | carla | Local-only; does not run CARLA, needs no network or simulator, and never asserts metric parity. | [dev_runtime_requirements.md](dev_runtime_requirements.md) | available |
| `robot-sf-carla-replay-diagnostics` | Build conservative CARLA replay diagnostics from Robot-SF and CARLA JSON inputs. | carla | Local-only diagnostics over supplied JSON files; no CARLA server, network, scheduler, or artifact mutation for `--help`. | [dev_runtime_requirements.md](dev_runtime_requirements.md) | available |
| `robot-sf-carla-t1-oracle-smoke` | Prepare one CARLA T1 oracle replay smoke from a T0 export manifest. | carla | Setup is local-only; live replay needs the external CARLA 0.9.16 Docker runtime and the `carla` client group. | [dev_runtime_requirements.md](dev_runtime_requirements.md) | available |
| `robot-sf-catalog-carla-schemas` | Print CARLA bridge schema catalog metadata. | carla | Local-only; no CARLA server, network, or simulator required. | [dev_runtime_requirements.md](dev_runtime_requirements.md) | available |
| `robot-sf-check-carla` | Check optional CARLA Python API availability. | carla | Local-only check; reports `not-available` when the `carla` client group is absent and never requires the simulator for `--help`. | [dev_runtime_requirements.md](dev_runtime_requirements.md) | available |
| `robot-sf-export-carla-t0` | Export Robot-SF scenarios to CARLA T0 neutral JSON. | carla | Help and `--schema` are local-only; export reads local scenario files only and needs no CARLA server or network. | [dev_runtime_requirements.md](dev_runtime_requirements.md) | available |
| `robot-sf-migrate-artifacts` | Consolidate legacy artifacts under the canonical `output/` root. | core | Always available (core dependencies only); `--help` and `--dry-run` are read-only, migration writes only under `output/` when actually run. | [dev_guide_reference.md](dev_guide_reference.md) | available |
| `robot-sf-validate-carla-t0-batch` | Validate a CARLA T0 export manifest and every referenced payload. | carla | Local-only validation; no CARLA server, network, or simulator required. | [dev_runtime_requirements.md](dev_runtime_requirements.md) | available |
| `robot-sf-validate-carla-t0-manifest` | Validate a CARLA T0 export manifest. | carla | Local-only validation; no CARLA server, network, or simulator required. | [dev_runtime_requirements.md](dev_runtime_requirements.md) | available |
| `robot_sf_bench` | Social-navigation benchmark toolkit (run, aggregate, validate, plot, SNQI, doctor). | benchmark | Requires the `benchmark` extra (included in `uv sync --all-extras`); `--help` is local-only with no network, simulator, scheduler, or artifact mutation. | [benchmark.md](benchmark.md) | available |

## Commands

### `robot-sf`

- Callable: `robot_sf.cli:main`
- Profile: `core`
- Availability: Always available after `uv sync --all-extras` (core dependencies only); `--help` is local-only with no network, simulator, scheduler, or artifact mutation.
- Guide: [adoption_path.md](adoption_path.md)
- Help: available
- Synopsis: Robot SF top-level command line interface.

Nested `robot-sf` subcommands (summary only; see each task guide for flags):

| Subcommand | Help | Guide |
| --- | --- | --- |
| `datasets` | List, prepare, and verify external datasets (uv run robot-sf datasets ...) | [external_data_setup.md](external_data_setup.md) |
| `demo` | Run the one-command visual demo (tiny deterministic episode + viewer). | [adoption_path.md](adoption_path.md) |
| `doctor` | Environment/readiness check with friendly remedies. | [adoption_path.md](adoption_path.md) |
| `envs` | List and describe registered public environments (uv run robot-sf envs ...) | [ENVIRONMENT.md](ENVIRONMENT.md) |
| `examples` | List and run examples from examples_manifest.yaml (issue #5794) | [adoption_path.md](adoption_path.md) |
| `gallery` | Build a static scenario/planner gallery (issue #5796) | [gallery.md](gallery.md) |
| `models` | List, download, and verify registered model artifacts (uv run robot-sf models ...) | [model_registry_publication.md](model_registry_publication.md) |
| `planners` | List and describe available planners (uv run robot-sf planners ...) | [planner_zoo/index.md](planner_zoo/index.md) |
| `recipe` | Curated recipe catalog (uv run robot-sf recipe list|run|explain) | [recipes/README.md](recipes/README.md) |
| `release` | Benchmark-data release operations. | [benchmark_artifact_publication.md](benchmark_artifact_publication.md) |
| `scenarios` | Discover, inspect, and validate scenarios (uv run robot-sf scenarios ...) | [scenario_zoo/index.md](scenario_zoo/index.md) |

### `robot-sf-carla-docker-runtime`

- Callable: `robot_sf_carla_bridge.cli:carla_docker_runtime_main`
- Profile: `carla`
- Availability: Preflight is host-local; `smoke` and `live-replay` need the Docker daemon, the pinned `carlasim/carla:0.9.16` image, and the `carla` client group; `--help` is local-only.
- Guide: [dev_runtime_requirements.md](dev_runtime_requirements.md)
- Help: available
- Synopsis: Preflight or smoke-test the pinned CARLA 0.9.16 Docker runtime.

Subcommands (summary only; see the task guide for flags):

| Subcommand | Guide |
| --- | --- |
| `live-replay` | [dev_runtime_requirements.md](dev_runtime_requirements.md) |
| `preflight` | [dev_runtime_requirements.md](dev_runtime_requirements.md) |
| `smoke` | [dev_runtime_requirements.md](dev_runtime_requirements.md) |

### `robot-sf-carla-parity-bundle-preflight`

- Callable: `robot_sf_carla_bridge.cli:preflight_carla_parity_bundle_main`
- Profile: `carla`
- Availability: Local-only; does not run CARLA, needs no network or simulator, and never asserts metric parity.
- Guide: [dev_runtime_requirements.md](dev_runtime_requirements.md)
- Help: available
- Synopsis: Read-only readiness preflight for the compact CARLA native/aligned parity bundle (#1510). Does not run CARLA or assert metric parity.

### `robot-sf-carla-replay-diagnostics`

- Callable: `scripts.carla_bridge.diagnose_replay_semantics:main`
- Profile: `carla`
- Availability: Local-only diagnostics over supplied JSON files; no CARLA server, network, scheduler, or artifact mutation for `--help`.
- Guide: [dev_runtime_requirements.md](dev_runtime_requirements.md)
- Help: available
- Synopsis: Build conservative CARLA replay diagnostics from Robot-SF and CARLA JSON inputs.

### `robot-sf-carla-t1-oracle-smoke`

- Callable: `robot_sf_carla_bridge.cli:replay_t1_oracle_smoke_main`
- Profile: `carla`
- Availability: Setup is local-only; live replay needs the external CARLA 0.9.16 Docker runtime and the `carla` client group.
- Guide: [dev_runtime_requirements.md](dev_runtime_requirements.md)
- Help: available
- Synopsis: Prepare one CARLA T1 oracle replay smoke from a T0 export manifest.

### `robot-sf-catalog-carla-schemas`

- Callable: `robot_sf_carla_bridge.cli:catalog_carla_schemas_main`
- Profile: `carla`
- Availability: Local-only; no CARLA server, network, or simulator required.
- Guide: [dev_runtime_requirements.md](dev_runtime_requirements.md)
- Help: available
- Synopsis: Print CARLA bridge schema catalog metadata.

### `robot-sf-check-carla`

- Callable: `robot_sf_carla_bridge.cli:check_carla_availability_main`
- Profile: `carla`
- Availability: Local-only check; reports `not-available` when the `carla` client group is absent and never requires the simulator for `--help`.
- Guide: [dev_runtime_requirements.md](dev_runtime_requirements.md)
- Help: available
- Synopsis: Check optional CARLA Python API availability.

### `robot-sf-export-carla-t0`

- Callable: `robot_sf_carla_bridge.cli:export_t0_scenarios_main`
- Profile: `carla`
- Availability: Help and `--schema` are local-only; export reads local scenario files only and needs no CARLA server or network.
- Guide: [dev_runtime_requirements.md](dev_runtime_requirements.md)
- Help: available
- Synopsis: Export Robot-SF scenarios to CARLA T0 JSON.

### `robot-sf-migrate-artifacts`

- Callable: `scripts.tools.migrate_artifacts:main`
- Profile: `core`
- Availability: Always available (core dependencies only); `--help` and `--dry-run` are read-only, migration writes only under `output/` when actually run.
- Guide: [dev_guide_reference.md](dev_guide_reference.md)
- Help: available
- Synopsis: Migration helper to consolidate legacy artifacts under the canonical `output/` root.

### `robot-sf-validate-carla-t0-batch`

- Callable: `robot_sf_carla_bridge.cli:validate_t0_export_batch_main`
- Profile: `carla`
- Availability: Local-only validation; no CARLA server, network, or simulator required.
- Guide: [dev_runtime_requirements.md](dev_runtime_requirements.md)
- Help: available
- Synopsis: Validate a CARLA T0 export batch.

### `robot-sf-validate-carla-t0-manifest`

- Callable: `robot_sf_carla_bridge.cli:validate_t0_manifest_main`
- Profile: `carla`
- Availability: Local-only validation; no CARLA server, network, or simulator required.
- Guide: [dev_runtime_requirements.md](dev_runtime_requirements.md)
- Help: available
- Synopsis: Validate a CARLA T0 export manifest.

### `robot_sf_bench`

- Callable: `robot_sf.benchmark.cli:main`
- Profile: `benchmark`
- Availability: Requires the `benchmark` extra (included in `uv sync --all-extras`); `--help` is local-only with no network, simulator, scheduler, or artifact mutation.
- Guide: [benchmark.md](benchmark.md)
- Help: available
- Synopsis: Social Navigation Benchmark CLI

Subcommands (summary only; see the task guide for flags):

| Subcommand | Guide |
| --- | --- |
| `admit-cases` | [benchmark.md](benchmark.md) |
| `aggregate` | [benchmark.md](benchmark.md) |
| `analyze-cases` | [benchmark.md](benchmark.md) |
| `baseline` | [benchmark.md](benchmark.md) |
| `claim` | [benchmark.md](benchmark.md) |
| `classify-failure-mechanisms` | [benchmark.md](benchmark.md) |
| `collision-scenario-similarity` | [benchmark.md](benchmark.md) |
| `debug-seeds` | [benchmark.md](benchmark.md) |
| `doctor` | [benchmark.md](benchmark.md) |
| `export-canonical-table` | [benchmark.md](benchmark.md) |
| `export-parquet` | [benchmark.md](benchmark.md) |
| `extract-failures` | [benchmark.md](benchmark.md) |
| `flakiness-audit` | [benchmark.md](benchmark.md) |
| `list-algorithms` | [benchmark.md](benchmark.md) |
| `list-scenarios` | [benchmark.md](benchmark.md) |
| `mapf-oracle` | [benchmark.md](benchmark.md) |
| `metric-layers` | [benchmark.md](benchmark.md) |
| `planner-inclusion-check` | [benchmark.md](benchmark.md) |
| `plot-distributions` | [benchmark.md](benchmark.md) |
| `plot-pareto` | [benchmark.md](benchmark.md) |
| `plot-planner-tradeoff` | [benchmark.md](benchmark.md) |
| `plot-scenarios` | [benchmark.md](benchmark.md) |
| `preview-scenarios` | [benchmark.md](benchmark.md) |
| `rank` | [benchmark.md](benchmark.md) |
| `run` | [benchmark.md](benchmark.md) |
| `seed-variance` | [benchmark.md](benchmark.md) |
| `snqi` | [benchmark.md](benchmark.md) |
| `snqi-ablate` | [benchmark.md](benchmark.md) |
| `stress-coverage-report` | [benchmark.md](benchmark.md) |
| `summary` | [benchmark.md](benchmark.md) |
| `table` | [benchmark.md](benchmark.md) |
| `validate-config` | [benchmark.md](benchmark.md) |
| `validate-row-claims` | [benchmark.md](benchmark.md) |

## Reproducibility

- Script order is sorted from `[project.scripts]` for determinism.
- `--help` runs in an isolated subprocess with a fixed width and timeout.
- No network, simulator, scheduler, or artifact mutation occurs during checks.
- CI fails on drift: run the generator without `--check` to refresh this file.
