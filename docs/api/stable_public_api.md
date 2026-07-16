# Stable public-API boundary & stability policy

This page documents the **supported public surface** of Robot SF and the
**stability policy** that governs it. It is the maintainer-review proposal for
issue #5801: it names what is *in* and *out* of the stable boundary and the
deprecation policy, but it is a **proposal** pending maintainer sign-off before
any enforcement (freezing `__all__`, adding `DeprecationWarning`s, or removing
internal-only exports) is performed.

## Stability levels

| Level | Meaning | Breaking-change rule |
| --- | --- | --- |
| `stable` | Supported public API. | Breaking changes follow semver `MAJOR` and ship with a `DeprecationWarning` one minor release before removal. |
| `beta` | Usable but may change | May change within a minor (`MINOR`) release **without** a deprecation window. |
| `experimental` | Exploratory. | Semantics may change at any time, including within a patch release. |

## Public surface (proposed)

### CLI (`robot-sf` top-level entry point)

Stable: `doctor`, `models list|verify|download`, `datasets list|verify|prepare`,
`demo`, `examples list|run`.

Beta (issue #5801): `envs list`, `envs describe <env-id>`.

### Python package — supported modules

* `robot_sf.gym_env.environment_factory` — typed `make_*` factories:
  `make_robot_env`, `make_image_robot_env`, `make_pedestrian_env`,
  `make_crowd_sim_env`, `make_multi_robot_env`.
* `robot_sf.gym_env.env_registry` — `list_envs`, `describe_env`, `get_env`,
  `register_env`, `env_ids`, and the `EnvEntry` dataclass (the catalog behind
  `robot-sf envs`).
* `robot_sf.sim.registry` — `register_backend`, `get_backend`, `list_backends`,
  `select_best_backend`.
* `robot_sf.sensor.registry` — `register_sensor`, `get_sensor`, `list_sensors`.
* `robot_sf.unified_config` — config dataclasses
  (`RobotSimulationConfig`, `ImageRobotConfig`, `PedestrianSimulationConfig`,
  `CrowdSimulationConfig`, `MultiRobotConfig`).
* `robot_sf.telemetry` — run tracking exports (`RunRegistry`,
  `ManifestWriter`, `RunTrackerConfig`, `generate_run_id`).

### Python package — internal / *not* in the stable boundary

These are implementation details. Importing them may break without notice
(they are not covered by the deprecation policy):

* `robot_sf.gym_env.base_env`, `_stub_robot_model`, and `EnvironmentFactory`
  (explicitly documented as *not exported*).
* Concrete environment classes under `robot_sf.gym_env.*_env` except via the
  `make_*` factories.
* Backend internals under `robot_sf.sim.backends.*`.
* Any `_`-prefixed name anywhere in the package.

## Deprecation policy (proposed)

1. A symbol leaving `stable` is first marked `beta` (or removed) with a
   `DeprecationWarning` and a `CHANGELOG.md` note.
2. Removal happens no earlier than one minor release after the warning.
3. `beta`/`experimental` symbols carry no removal guarantee.

## Discover the live environment catalog

```bash
uv run robot-sf envs list
uv run robot-sf envs describe <env-id>
```

The catalog is the declarative source of truth for which environment ids are
public; see `robot_sf.gym_env.env_registry` for the registry implementation.
