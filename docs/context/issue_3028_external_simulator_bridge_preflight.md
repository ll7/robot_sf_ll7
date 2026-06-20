# Issue #3028 External Simulator Bridge Preflight

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/3028>

**Scope**: preflight / scoping note. Evidence tier: `idea`. No realism claim, no benchmark claim,
no paper-facing evidence. Out of scope: full CARLA or Unity support, core sim-API changes,
perception integration, sensor-realism benchmarks.

## 1. Claim Boundary

This note scopes the gap between what the CARLA bridge already delivers (offline export + replay +
oracle parity) and what a live, closed-loop, online sensor-streaming bridge would require. It
defines an abstract contract against existing extension points and lists blocked prerequisites. No
code changes accompany this note.

## 2. Current Support Inventory

A substantial offline CARLA bridge already exists under `robot_sf_carla_bridge/`. The package is
import-safe without CARLA installed (`robot_sf_carla_bridge/__init__.py` uses `require_carla()` in
`availability.py` as a lazy guard). The pinned runtime dependency is
`carla==0.9.16 ; sys_platform == 'linux' and platform_machine == 'x86_64'` (`pyproject.toml`,
`[dependency-groups.carla]`), making the Python API linux/x86_64-only and intentionally excluded
from routine `uv sync`.

### Offline export path (T0)

- `robot_sf_carla_bridge/export.py` — `build_export_payloads_from_scenario_file`, `write_export_records`,
  `validate_export_payload`: convert Robot-SF scenario manifests to CARLA-neutral T0 JSON payloads.
- `robot_sf_carla_bridge/schema_catalog.py` — `list_carla_bridge_schema_catalog`: catalogs all
  bridge JSON schemas (`schemas/carla_replay_export.v1.json`, `carla_replay_export_manifest.v1.json`,
  etc.) with version pinning.

### Runtime availability and Docker preflight

- `robot_sf_carla_bridge/availability.py` — `check_carla_availability`, `require_carla`: explicit
  availability metadata; uses `importlib.util.find_spec` so no import-time CARLA dependency.
- `robot_sf_carla_bridge/docker_runtime.py` — `run_carla_docker_preflight`, `run_carla_docker_runtime_smoke`,
  `run_carla_docker_live_replay`, `build_carla_server_container_command`,
  `validate_carla_image`: full Docker image validation and container lifecycle.
- `scripts/dev/check_carla_runtime.sh` — host-side preflight: runs `robot-sf-check-carla --json`
  then `robot-sf-carla-docker-runtime {preflight|smoke}`; supports `--pull` and
  `--startup-timeout-s`.

### T1 oracle replay

- `robot_sf_carla_bridge/replay_smoke.py` — `build_t1_oracle_replay_smoke_setup`,
  `select_t0_export_payload`, `validate_t1_replay_catalog_payload`: setup-only boundary before
  live CARLA connection.
- `robot_sf_carla_bridge/live_replay.py` — `run_t1_oracle_live_replay_against_server`,
  `robot_sf_pose_to_carla_transform`: live oracle replay that teleports actors to oracle poses
  frame-by-frame via CARLA's `actor.set_transform()`; includes coordinate conversion (Robot-SF
  right-handed x/y → CARLA/Unreal with y and yaw negated).

### Diagnostics and parity

- `robot_sf_carla_bridge/diagnostics.py` — `build_carla_replay_diagnostics`,
  `write_carla_replay_diagnostics_outputs`: conservative row-level diagnostics for replay traces.
- `robot_sf_carla_bridge/parity.py` — `compare_oracle_replay_metrics` against
  `DEFAULT_PARITY_METRICS` (success, collision, ttc_min_s, min_distance_m, comfort, jerk,
  curvature, intervention_rate, snqi).
- `scripts/carla_bridge/diagnose_replay_semantics.py` — CLI wrapper for
  `build_carla_replay_diagnostics`.
- `scripts/carla_bridge/compare_oracle_replay_metrics.py` — CLI wrapper for
  `compare_oracle_replay_metrics`.

### CLI entry points (pyproject.toml, lines 486-493)

| Script | Entry point |
|---|---|
| `robot-sf-export-carla-t0` | `export_t0_scenarios_main` |
| `robot-sf-validate-carla-t0-manifest` | `validate_t0_manifest_main` |
| `robot-sf-validate-carla-t0-batch` | `validate_t0_export_batch_main` |
| `robot-sf-check-carla` | `check_carla_availability_main` |
| `robot-sf-catalog-carla-schemas` | `catalog_carla_schemas_main` |
| `robot-sf-carla-t1-oracle-smoke` | `replay_t1_oracle_smoke_main` |
| `robot-sf-carla-docker-runtime` | `carla_docker_runtime_main` |
| `robot-sf-carla-replay-diagnostics` | `scripts.carla_bridge.diagnose_replay_semantics:main` |

## 3. Gap Analysis

What is NOT yet present:

1. **No live closed-loop stepping bridge.** The existing `live_replay.py` replay is oracle-driven:
   it teleports pre-computed actor poses, step by step, from a frozen T0 replay. It does not
   stream CARLA sensor observations back into robot_sf's policy/environment loop; the robot's
   policy receives no CARLA-rendered sensor data. A true closed-loop bridge would need to (a) step
   the CARLA world in synchronous mode, (b) attach sensors (camera, LiDAR, radar) to CARLA actors,
   (c) read sensor data per tick, (d) convert it to robot_sf `Sensor` observations, and (e)
   forward the resulting action back to CARLA.

2. **No `SimulatorFactory` registration for CARLA.** `robot_sf/sim/registry.py` currently holds
   `fast-pysf` and `dummy` backends (`register_backend`, `get_backend`, `list_backends`). No
   `carla` backend is registered. `robot_sf/sim/facade.py` defines `SimulatorFactory` as
   `Callable[[EnvSettings, MapDefinition, bool], Any]`; a CARLA backend would need to satisfy this
   signature while holding a CARLA world handle and synchronous-mode context.

3. **No sensor-registry bridge.** `robot_sf/sensor/registry.py` (`register_sensor`, `get_sensor`,
   `list_sensors`) and `robot_sf/sensor/base.py` (the `Sensor` Protocol: `reset`, `step`, `get_observation`)
   provide clean extension hooks, but no CARLA-backed sensor implementations are registered. The
   `step(state)` call presently receives a robot_sf internal state object, not a CARLA sensor
   buffer.

4. **No Unity path.** No Unity runtime, SDK dependency, or bridge module exists anywhere in the
   repository. Unity would require a separate runtime process, communication protocol (e.g. ML-Agents
   or a custom socket bridge), and licensing/toolchain path entirely distinct from CARLA's Python API.

5. **No synchronization contract.** CARLA's synchronous mode requires explicit `world.tick()` calls.
   robot_sf's environment loop does not currently manage an external world clock. Coordinating tick
   timing, sensor-data latency, and episode reset across a socket boundary is not addressed.

## 4. Abstract Bridge Contract

The following describes the interface design in prose/pseudocode. No code is added to the
repository by this note.

### 4.1 SimulatorFactory backend ("carla")

A CARLA backend factory registered under the key `"carla"` via `register_backend` in
`robot_sf/sim/registry.py` would accept `(env_config, map_def, peds_have_obstacle_forces)` and
return an object satisfying the simulator contract (the attributes used by `RobotEnv`: `robots`,
`robot_poses`, `ped_pos`, `goal_pos`, `step_once(actions)`, `reset_state()`, `map_def`,
`get_obstacle_lines()`).

Internally the CARLA backend would:
- connect to a running CARLA server via the Python API (`carla.Client(host, port)`);
- load a map compatible with a T0 export payload;
- spawn ego vehicle and pedestrian actors;
- enable synchronous mode (`world.apply_settings(synchronous_mode=True, fixed_delta_seconds=dt)`);
- on `step_once(actions)`, apply the action to the ego actor, call `world.tick()`, and collect
  updated actor state.

Availability guard: the factory module must use `require_carla()` from
`robot_sf_carla_bridge/availability.py` so the registry import does not hard-require CARLA.

### 4.2 Sensor bridge via the Sensor Protocol

Each CARLA sensor (e.g., camera, LiDAR) would be implemented as a class satisfying
`robot_sf/sensor/base.py:Sensor` (Protocol):
- `reset() -> None`: destroy and respawn the CARLA sensor actor; flush pending data.
- `step(state) -> None`: in synchronous mode, data is already available after `world.tick()`; this
  method reads the sensor queue and stores the converted observation.
- `get_observation() -> Any`: return the most recent converted sensor tensor/array in a shape
  compatible with the environment observation space.

Each such sensor would be registered with `register_sensor("carla_lidar", carla_lidar_factory)`
etc. in `robot_sf/sensor/registry.py`.

### 4.3 Synchronization contract

The CARLA backend's `step_once` is the single synchronization point:
1. Apply ego action to CARLA vehicle.
2. Call `world.tick()` — CARLA advances one fixed timestep.
3. All attached sensors deliver data for this tick via their listen callbacks.
4. Each sensor's `step(state)` is invoked to consume the callback data.
5. The environment reads `get_observation()` per sensor and constructs the RL observation.

Episode reset calls `reset_state()` on the backend (respawn actors, clear sensor queues) and
`reset()` on each sensor. Connection teardown on `__del__` or explicit `close()` should restore
asynchronous mode and destroy spawned actors.

## 5. Host and Platform Constraints

- **Platform**: CARLA 0.9.16 is linux/x86_64-only (`pyproject.toml`). macOS and Windows CI
  runners cannot run the CARLA Python API; all CARLA-dependent tests must be conditioned on
  `check_carla_availability()["available"]`.
- **Docker image**: `docker_runtime.py` pins a specific CARLA Docker image (`CARLA_DOCKER_IMAGE`).
  Running the simulator requires Docker with GPU passthrough (NVIDIA runtime). The startup timeout
  is configurable via `--startup-timeout-s` in `check_carla_runtime.sh`.
- **GPU requirement**: CARLA's renderer requires a GPU; headless/CPU-only CI hosts cannot run a
  live CARLA world. Sensor-data tests need a GPU host or a no-rendering mock.
- **Unity**: No platform path scoped. Unity ML-Agents or a custom bridge would require a separate
  licensing and toolchain evaluation that is entirely out of scope for this note.

## 6. Blocked Prerequisites and Next Steps

### Blocked prerequisites

| Prerequisite | Blocker |
|---|---|
| Live sensor bridge | No CARLA backend registered in `robot_sf/sim/registry.py` |
| Sensor data contract | `Sensor.step(state)` currently takes robot_sf state; CARLA sensors deliver raw tensors via callbacks; adapter shape is unspecified |
| Synchronous-mode integration | `RobotEnv` does not manage an external world clock; tick ownership is unresolved |
| CI coverage | No GPU host in current CI; CARLA-dependent live tests cannot run in standard pipeline |
| Unity | No runtime path, SDK, or protocol specified |

### Smallest next step

Review and ratify the abstract bridge contract in Section 4 with the maintainer before writing
any code. Specifically:

1. Decide whether `Sensor.step(state)` signature should be extended to carry a CARLA tick token,
   or whether sensor read-back is done inside the backend's own `step_once`.
2. Confirm which map/actor spawning conventions are compatible with existing T0 export payloads.
3. Identify a GPU-enabled host or mock strategy for CI.

### Decision / stop rule

Keep this note design-note-only until the live-sensor contract review is complete. Do not
implement the CARLA backend or sensor adapters until:
- the `Sensor.step` interface question is resolved;
- a GPU test host or no-rendering mock is confirmed;
- at least one T0 export payload has been validated against a live CARLA world with `run_t1_oracle_live_replay_against_server`.

If the interface review surfaces fundamental incompatibilities (e.g. tick ownership cannot be
cleanly separated from `RobotEnv`), re-scope as a backend-adapter contract issue first
(see `docs/context/issue_2013_backend_adapter_contract.md`) before attempting a sensor bridge.
Unity scoping should wait for a separate dedicated issue if CARLA integration is confirmed viable.
