# Robot SF Public API

This document describes the supported public API surface of Robot SF, its stability levels, and its lifecycle guarantees.

Robot SF provides a lightweight top-level facade for scenario-based social navigation simulation, alongside modular subpackages for environment creation, simulation backends, sensors, and telemetry tracking.

## Stability Levels

Robot SF uses three explicit stability levels to communicate breaking-change expectations across releases:

| Level | Meaning | Breaking-Change Rule |
| --- | --- | --- |
| `stable` | Supported public API. | Breaking changes follow semantic versioning (`MAJOR`) and ship with a `DeprecationWarning` at least two minor releases before removal. |
| `beta` | Supported and usable, but undergoing refinement. | May change within a minor (`MINOR`) release without an extended deprecation window; changes are documented in `CHANGELOG.md`. |
| `experimental` | Exploratory research surfaces. | Semantics may change at any time, including within a patch release. |

Environment configuration fields are documented separately in the [Environment
Configuration Reference](./environment_config_reference.md), generated from the typed
configuration dataclasses.

## Top-Level Entry Points

The `robot_sf` top-level package provides lightweight, lazily resolved exports for standard simulation workflows. Importing the top-level package (`import robot_sf`) is fast and does not eagerly import heavy visualization or machine learning dependencies (such as Pygame, PyTorch, or Stable-Baselines3).

All symbols in `robot_sf.__all__` are part of the `stable` public facade:

### Top-Level Facade Manifest

<!-- public-api-manifest:start -->
| Symbol | Stability | Kind | Source |
| --- | --- | --- | --- |
| `EpisodeRecord` | stable | class | `robot_sf.api` |
| `ManifestWriter` | stable | class | `robot_sf.telemetry` |
| `PlannerProtocol` | stable | protocol | `robot_sf.api` |
| `RunRegistry` | stable | class | `robot_sf.telemetry` |
| `RunTrackerConfig` | stable | class | `robot_sf.telemetry` |
| `ScenarioSpec` | stable | class | `robot_sf.api` |
| `api` | stable | module | `robot_sf.api` |
| `generate_run_id` | stable | function | `robot_sf.telemetry` |
| `load_scenario` | stable | function | `robot_sf.api` |
| `make_env` | stable | function | `robot_sf.api` |
| `run_episode` | stable | function | `robot_sf.api` |
| `telemetry` | stable | module | `robot_sf.telemetry` |
<!-- public-api-manifest:end -->

### Environment Creation

- **`robot_sf.make_env(*, scenario=None, seed=None, **kwargs)`**:
  Convenience keyword-only factory for creating robot simulation environments. When `scenario` is provided (as a path, scenario name, or mapping), simulation configuration is automatically loaded and resolved from `configs/scenarios/`. A mapping returned by `load_scenario` may be passed through unchanged; if a caller constructs a mapping independently, its `map_file` and `route_overrides_file` references must be absolute unless the source metadata from `load_scenario` is preserved. Relative asset paths without that source metadata are rejected instead of being resolved from an arbitrary working directory. An explicit `scenario_name` is preserved as the environment identity. All additional options are forwarded to `robot_sf.gym_env.environment_factory.make_robot_env`.

### Scenario Resolution

- **`robot_sf.load_scenario(scenario_id)`**:
  Resolves and parses scenario YAML definitions from `configs/scenarios/`. Accepts file paths, paths relative to `configs/scenarios/`, or scenario stems/names. Identifier lookup prefers canonical `single/` and `archetypes/` definitions over aggregate manifests, deduplicates equivalent re-exports, and still raises when conflicting definitions remain in the selected source class. Source checkouts provide the canonical scenario tree; an installed package without that asset tree fails closed with an actionable `FileNotFoundError`, so callers in that deployment mode must pass an explicit scenario file path.

### Episode Execution

- **`robot_sf.run_episode(env, *, planner=None, max_steps=None, seed=None)`**:
  Executes a single seeded episode on the provided Gymnasium environment, stepping the optional planner (conforming to `PlannerProtocol`) or default actions, and returns an `EpisodeRecord`. Built-in baseline planners with the canonical observation contract receive a world-frame `Observation` including static obstacle segments; callable-only, custom step-method, and explicitly dict-native planners receive the raw environment observation. `max_steps`, when supplied, must be a positive integer. Planner objects must provide a callable `step()` method or themselves be callable; invalid objects are rejected rather than replaced with random actions. Episode IDs follow the repository's stable `<scenario>--<seed>` identity convention.

### Core Data Structures and Protocols

- **`robot_sf.EpisodeRecord`**:
  Dataclass capturing episode metrics, horizon, seed, algorithm name, and execution timing. Supports JSON persistence via `record.save(path)` and deserialization via `EpisodeRecord.load(path)`.
- **`robot_sf.ScenarioSpec`**:
  Dataclass representing scenario specifications within scenario matrices.
- **`robot_sf.PlannerProtocol`**:
  Protocol defining the standard interface for navigation planners (`step`, `reset`, `configure`, `close`).
- **`robot_sf.api`**:
  Module re-exporting the public facade functions and types. For complete runnable first-episode examples and exception contracts, see the [Sphinx API Facade Reference](api/robot_sf.api.rst).

### Telemetry Surface

The following telemetry utilities are exported at top-level:

- `robot_sf.ManifestWriter`: Structured telemetry manifest writer.
- `robot_sf.RunRegistry`: Registry recording experiment and benchmark runs.
- `robot_sf.RunTrackerConfig`: Configuration dataclass for run telemetry.
- `robot_sf.generate_run_id`: Deterministic or random run identifier generator.
- `robot_sf.telemetry`: Telemetry submodule.

## CLI Public Surface (`robot-sf`)

The `robot-sf` command line interface provides contributor and user tooling:

- **Stable CLI commands**:
  - `doctor`: Inspect environment dependencies, hardware, and configuration health.
  - `models`: `list`, `verify`, `download` model checkpoints.
  - `datasets`: `list`, `verify`, `prepare` benchmark datasets.
  - `demo`: Interactive demonstration launcher.
  - `examples`: `list`, `run` bundled examples.
  - `recipe`: `list`, `explain`, `run` curated workflows defined by the repository recipe catalog.
- **Beta CLI commands**:
  - `envs`: `list`, `describe <env-id>` for declarative environment inspection.
- **Experimental CLI commands**:
  - `planners`: `list`, `describe <key-or-alias>` for import-light planner and baseline discovery.
  - `gallery`: `build` discoverability-only scenario/planner galleries; gallery output is not
    benchmark evidence.
- **Maintainer/release CLI command**:
  - `release`: benchmark-data release operations, including the read-only `audit-published`,
    fail-closed `doctor`, and guarded `zenodo` workflows. See [Release Operations](./RELEASE.md);
    this maintainer-facing command is not part of the stable end-user API.

## Supported Python Modules Catalog

Beyond the top-level facade, specific subpackages provide supported modular APIs:

- **`robot_sf.gym_env.environment_factory`**:
  Typed factory functions: `make_robot_env`, `make_image_robot_env`, `make_pedestrian_env`, `make_crowd_sim_env`, `make_multi_robot_env`.
- **`robot_sf.gym_env.env_registry`**:
  Environment catalog registry: `list_envs`, `describe_env`, `get_env`, `register_env`, `env_ids`, and the `EnvEntry` dataclass.
- **`robot_sf.sim.registry`**:
  Simulation backend registry: `register_backend`, `get_backend`, `list_backends`, `select_best_backend`.
- **`robot_sf.sensor.registry`**:
  Sensor registry: `register_sensor`, `get_sensor`, `list_sensors`.
- **`robot_sf.gym_env.unified_config`**:
  Typed configuration dataclasses: `RobotSimulationConfig`, `ImageRobotConfig`, `PedestrianSimulationConfig`, `MultiRobotConfig`.
- **`robot_sf.gym_env.crowd_sim_env`**:
  `CrowdSimulationConfig`.
- **`robot_sf.telemetry`**:
  Run tracking infrastructure (`RunRegistry`, `ManifestWriter`, `RunTrackerConfig`, `generate_run_id`).

## Internal Architecture Boundaries

The following modules and patterns are implementation details and are **not** part of the stable public API. Importing or relying on them directly may break between minor or patch releases:

- `robot_sf.gym_env.base_env`, `_stub_robot_model`, and `EnvironmentFactory` (callers must use the `make_*` factory functions).
- Concrete environment subclasses under `robot_sf.gym_env.*_env` directly (instantiate via factories).
- Backend internals under `robot_sf.sim.backends.*`.
- Any private symbol (prefixed with `_`) across the entire codebase.

## Discovering the Live Environment Catalog

To inspect available environment configurations interactively:

```bash
uv run robot-sf envs list
uv run robot-sf envs describe <env-id>
```

The catalog is the declarative source of truth for public environment identifiers; see `robot_sf.gym_env.env_registry` for programmatic access.

## Discovering the Planner Catalog

To inspect registered planners, aliases, readiness tiers, and declared prerequisites without
constructing planner instances or importing optional heavy dependencies:

```bash
uv run robot-sf planners list
uv run robot-sf planners describe <key-or-alias>
```

Discovery metadata is not benchmark-success evidence. An `unknown` availability value means that
local optional dependencies were not probed; diagnostic, unavailable, and metadata-incomplete rows
remain separate from claims that a planner is runnable or benchmark-ready.

## Lifecycle and Deprecation Policy

### 1. Environment Lifecycle

- Environments implement the standard Gymnasium lifecycle:
  ```python
  obs, info = env.reset(seed=seed)
  obs, reward, terminated, truncated, info = env.step(action)
  env.close()
  ```
- `env.close()` releases rendering views, recorders, and underlying simulation resources idempotently. Calling `close()` multiple times is safe.
- Legacy `env.exit()` is deprecated and issues a `DeprecationWarning`.

### 2. Deprecation Window

- A symbol leaving `stable` status is first marked with a `DeprecationWarning` and documented in `CHANGELOG.md`.
- Deprecated symbols, functions, arguments, and aliases are maintained for a minimum of two minor releases before removal.
- `beta` and `experimental` symbols carry no long-term removal guarantee, though breaking changes are noted in `CHANGELOG.md`.
