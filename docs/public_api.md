# Robot SF Public API

This document describes the supported top-level public API surface of Robot SF and its lifecycle guarantees.

## Top-Level Entry Points

The `robot_sf` package provides lightweight, lazily resolved top-level exports for common simulation workflows. Package import (`import robot_sf`) does not eagerly import heavy visualization or learning dependencies (such as Pygame, PyTorch, or Stable-Baselines3).

### Environment Creation

- **`robot_sf.make_env(*, scenario=None, seed=None, **kwargs)`**:
  Convenience keyword-only factory for creating robot simulation environments. When `scenario` is provided (as a path, scenario name, or mapping), simulation configuration is automatically loaded and resolved from `configs/scenarios/`. All additional options are forwarded to `robot_sf.gym_env.environment_factory.make_robot_env`.

### Scenario Resolution

- **`robot_sf.load_scenario(scenario_id)`**:
  Resolves and parses scenario YAML definitions from `configs/scenarios/`. Accepts file paths, paths relative to `configs/scenarios/`, or scenario stems/names.

### Episode Execution

- **`robot_sf.run_episode(env, *, planner=None, max_steps=None, seed=None)`**:
  Executes a single seeded episode on the provided Gymnasium environment, stepping the optional planner (conforming to `PlannerProtocol`) or default actions, and returns an `EpisodeRecord`.

### Core Data Structures and Protocols

- **`robot_sf.EpisodeRecord`**:
  Dataclass capturing episode metrics, horizon, seed, algorithm name, and execution timing. Supports JSON persistence via `record.save(path)` and deserialization via `EpisodeRecord.load(path)`.
- **`robot_sf.ScenarioSpec`**:
  Dataclass representing scenario specifications within scenario matrices.
- **`robot_sf.PlannerProtocol`**:
  Protocol defining the standard interface for navigation planners (`step`, `reset`, `configure`, `close`).

### Telemetry Surface

The following telemetry utilities remain available at top-level:

- `robot_sf.ManifestWriter`
- `robot_sf.RunRegistry`
- `robot_sf.RunTrackerConfig`
- `robot_sf.generate_run_id`
- `robot_sf.telemetry`

## Lifecycle and Deprecation Policy

1. **Environment Lifecycle**:
   - Environments implement the standard Gymnasium lifecycle: `obs, info = env.reset(seed=...)` and `env.close()`.
   - `env.close()` releases rendering views, recorders, and underlying simulation resources idempotently. Calling `close()` multiple times is safe.
   - Legacy `env.exit()` is deprecated and issues a `DeprecationWarning`.

2. **Deprecation Window**:
   - Deprecated functions, arguments, and aliases are maintained for a minimum of two minor releases with explicit warnings before removal.
