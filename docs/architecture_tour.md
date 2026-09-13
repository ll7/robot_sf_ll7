# Robot SF Architecture Tour

Robot SF is a Gymnasium-based social-navigation simulator. This page gives contributors one
source-linked path from a public API call to a returned episode record, then shows where common
changes belong. It is an orientation guide, not a benchmark claim or an exhaustive module inventory.

## One episode, end to end

The smallest useful mental model has two executable episode paths. The public API path builds an
environment and returns a typed `EpisodeRecord`; the benchmark path produces a richer dictionary
record and persists it only after validation.

```text
public API path
caller
  -> robot_sf.api.load_scenario()
  -> robot_sf.api.make_env()
       -> environment_factory.make_robot_env()
       -> RobotEnv / BaseEnv
       -> Simulator
  -> robot_sf.api.run_episode(env, planner=...)
       -> env.reset(seed=...)
            <- observation and reset info
       -> planner action adapter (step(...) or callable)
       -> env.step(action)
            -> simulator transition, reward, next observation, and step info
       -> EpisodeRecord

benchmark path
caller
  -> benchmark.runner.run_episode(scenario_params, seed)
       -> dict[str, Any] with trajectory, metrics, and provenance
  -> benchmark.runner.validate_and_write(...)
       -> validated benchmark JSONL record
  -> evidence writer or registered artifact output
```

### 1. Resolve a scenario

`robot_sf.api.load_scenario()` resolves a scenario identifier, file stem, or explicit path under
[`configs/scenarios/`](../configs/scenarios/). It loads the YAML through the scenario-loader owner
and keeps the source path attached to the returned mapping so relative map and route assets remain
unambiguous. A scenario definition describes inputs; it is not itself an executed episode.

### 2. Build the environment

`robot_sf.api.make_env()` is the ergonomic public facade. It converts a scenario mapping into the
typed configuration expected by [`environment_factory.py`](../robot_sf/gym_env/environment_factory.py)
and delegates construction to `make_robot_env()`.

The concrete [`RobotEnv`](../robot_sf/gym_env/robot_env.py) is the Gymnasium-facing environment.
Its shared [`BaseEnv`](../robot_sf/gym_env/base_env.py) owns common lifecycle, recording, and
rendering hooks. Environment construction creates the configured
[`Simulator`](../robot_sf/sim/simulator.py), which coordinates robot navigation, pedestrian
dynamics, collision detection, and timestep synchronization.

### 3. Exchange observations and actions

On the public API path, `robot_sf.api.run_episode()` owns the reset and step loop: it calls
`env.reset(seed=...)`, sends each observation through its planner action adapter, and passes the
result to `env.step(action)`. The environment advances the simulator and returns the next
observation, reward, termination flags, and step information. Do not manually step the environment
before calling `run_episode()`.

There are two deliberately distinct planner protocol families:

- Baseline planners use [`PlannerProtocol`](../robot_sf/baselines/interface.py), whose main
  action method is `step(obs) -> dict`.
- Native local-planner integrations use [`LocalPlannerProtocol`](../robot_sf/planner/protocol.py),
  whose action method is `plan(observation) -> (linear_speed, angular_rate)` at that integration
  boundary.

Adapters may bridge these families, but a bridge must keep missing diagnostics, dependency
availability, and fallback/degraded status explicit. A passing adapter test does not establish
planner quality or benchmark evidence.

### 4. Produce an episode record

For a lightweight public-API run, [`robot_sf.api.run_episode()`](../robot_sf/api.py) returns an
[`EpisodeRecord`](../robot_sf/benchmark/types.py) containing the episode identity, scenario,
seed, metrics, algorithm name, horizon, timing, and optional raw metadata.

The benchmark path has a separate, richer record contract. [`robot_sf.benchmark.runner`](../robot_sf/benchmark/runner.py)
provides `run_episode(scenario_params, seed) -> dict[str, Any]`, `run_batch()`, and
`validate_and_write()`. The runner captures trajectory/metric data, validates records against the
versioned episode schema, and appends valid records to JSONL. Do not use a lightweight API
`EpisodeRecord` as a substitute for a benchmark record or silently omit required provenance.

### 5. Write evidence or diagnostic artifacts

When a workflow produces a tracked diagnostic or evidence artifact, reuse the shared writers in
[`robot_sf/evidence/writers.py`](../robot_sf/evidence/writers.py). They provide deterministic
JSON/CSV/text output, review markers, checksums, and evidence-tree registration hooks. Keep raw
episodes, videos, checkpoints, and other large generated outputs in the worktree-local output
area unless a separate custody contract promotes them to durable storage.

### 6. Keep optional surfaces explicit

- Visualization and rendering use [`robot_sf/render/`](../robot_sf/render/) for frame capture,
  playback, and visual inspection. Rendering is an output/debug surface; a plot or video is not
  benchmark evidence unless its input, manifest, and provenance contract says so.
- Training and external-data workflows use [`robot_sf/training/`](../robot_sf/training/) and
  [`docs/external_data_setup.md`](external_data_setup.md). Checkpoint, dataset, license, and
  checksum gates remain separate from simulator execution; missing assets stay unavailable.
- CARLA integration uses [`robot_sf_carla_bridge/`](../robot_sf_carla_bridge/) and the
  [CARLA T0/T1 replay contract](context/issue_928_carla_t0_t1_replay_contract.md). The bridge is
  optional: the `carla` client group, pinned Docker runtime, server, map, and certified input are
  explicit prerequisites. Without them, availability must be reported as `not-available`; a CPU
  Robot-SF run is not a CARLA fallback, and an export alone is not parity evidence.

## Package map

The public supported entry points are the API facade and the Gymnasium environment protocol. The
package rows below identify canonical implementation owners; a source link is not by itself a
promise that every internal module is a stable public API.

| Concern | Canonical owner | What belongs there |
| --- | --- | --- |
| Public entry points | [`robot_sf/api.py`](../robot_sf/api.py) and [`environment_factory.py`](../robot_sf/gym_env/environment_factory.py) | Scenario loading, environment construction, and ergonomic episode execution. |
| Environment and sensors | [`robot_sf/gym_env/`](../robot_sf/gym_env/) | Gymnasium lifecycle, observations, rewards, recording, and sensor wiring. |
| Simulation and physics | [`robot_sf/sim/`](../robot_sf/sim/) and [`fast-pysf/`](../fast-pysf/) | Robot/pedestrian state transitions and force backends. Preserve site-specific and legacy contracts. |
| Planner protocols and adapters | [`robot_sf/planner/`](../robot_sf/planner/) and [`robot_sf/baselines/`](../robot_sf/baselines/) | Local planners, baseline planners, protocol adapters, and planner diagnostics. |
| Scenarios and maps | [`configs/scenarios/`](../configs/scenarios/) and [`maps/`](../maps/) | Reusable scenario inputs and authored map assets; validate paths and geometry before runtime use. |
| Benchmark records and metrics | [`robot_sf/benchmark/`](../robot_sf/benchmark/) | Versioned record schemas, metric definitions, campaign runners, and provenance checks. |
| Evidence and publication | [`robot_sf/evidence/`](../robot_sf/evidence/) and [`docs/context/evidence/`](context/evidence/) | Artifact writing, catalog registration, diagnostic reports, and bounded evidence handoff. |
| Visualization and rendering | [`robot_sf/render/`](../robot_sf/render/) | Frame capture, playback, simulation views, and visual debugging; keep optional dependencies explicit. |
| Training and external assets | [`robot_sf/training/`](../robot_sf/training/) and [`docs/external_data_setup.md`](external_data_setup.md) | Training workflows and license/provenance-safe external-data intake. |
| CARLA integration | [`robot_sf_carla_bridge/`](../robot_sf_carla_bridge/) and [the T0/T1 contract](context/issue_928_carla_t0_t1_replay_contract.md) | Optional scenario export, availability checks, oracle replay, and parity diagnostics; missing runtime support is not fallback success. |
| Documentation and examples | [`docs/`](./) and [`examples/`](../examples/) | User guidance, architecture/context notes, and reproducible usage examples. |

## Where common changes belong

Use the smallest owner that already represents the behavior. The validation boundary should follow
the changed contract, not just the file extension.

| Change | Start here | Minimum review boundary |
| --- | --- | --- |
| Public API or environment factory | [`robot_sf/api.py`](../robot_sf/api.py), [`robot_sf/gym_env/`](../robot_sf/gym_env/) | Focused API/environment tests, type/lint checks, and updated public docs. |
| Observation, action, or planner integration | [`robot_sf/baselines/interface.py`](../robot_sf/baselines/interface.py), [`robot_sf/planner/`](../robot_sf/planner/) | Protocol/adapter tests and explicit missing-dependency/fallback behavior. Domain review is required when behavior or benchmark interpretation changes. |
| Physics, collision, or pedestrian behavior | [`robot_sf/sim/`](../robot_sf/sim/) or [`fast-pysf/`](../fast-pysf/) | Focused runtime and parity tests; preserve legacy inputs and require domain-aware review for default-law or physical claims. |
| Metric, schema, or benchmark workflow | [`robot_sf/benchmark/`](../robot_sf/benchmark/) and [`configs/benchmarks/`](../configs/benchmarks/) | Contract tests, provenance/fallback checks, reproducible sample execution, and the full benchmark-sensitive readiness gate. |
| External model or dataset | [`model/`](../model/) and [`docs/external_data_setup.md`](external_data_setup.md) | Rights, source custody, checksum, and availability proof. Missing assets must remain unavailable rather than becoming fallback success. |
| Visualization, playback, or video | [`robot_sf/render/`](../robot_sf/render/) | Headless/render tests and optional-dependency checks; visual output alone is not benchmark evidence. |
| CARLA export or replay bridge | [`robot_sf_carla_bridge/`](../robot_sf_carla_bridge/) and [the T0/T1 contract](context/issue_928_carla_t0_t1_replay_contract.md) | CARLA-free import/availability/contract tests first; live replay needs the exact optional client/runtime boundary and cannot claim parity from export alone. |
| Documentation or example | [`docs/`](./) or [`examples/`](../examples/) | Link/path checks, focused docs or example smoke tests, and strict Sphinx validation when navigation changes. |

For terminology such as Gymnasium, Social Navigation Quality Index (SNQI), and vulnerable road
user (VRU), use the [project glossary](glossary.md). For reproducible research workflows, continue
with the [research and benchmark guide](research-guide.md), then consult the relevant benchmark
or evidence contract before interpreting a result.
