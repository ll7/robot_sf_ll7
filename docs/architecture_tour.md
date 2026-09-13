# Robot SF Architecture Tour

Robot SF is a Gymnasium-based social-navigation simulator. This page gives contributors one
source-linked path from a public API call to a returned episode record, then shows where common
changes belong. It is an orientation guide, not a benchmark claim or an exhaustive module inventory.

## One episode, end to end

The smallest useful mental model is:

```text
caller
  -> robot_sf.api.load_scenario()
  -> robot_sf.api.make_env()
       -> environment_factory.make_robot_env()
       -> RobotEnv / BaseEnv
       -> Simulator
  -> env.reset(seed=...)
       <- observation and reset info
  -> planner.step(...) or local-planner.plan(...)
  -> env.step(action)
       -> simulator transition, reward, next observation, and step info
  -> robot_sf.api.run_episode() or benchmark.runner.run_episode()
  -> EpisodeRecord / validated benchmark record
  -> evidence writer or JSONL output
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

An episode starts with `env.reset(seed=...)`, which returns an observation and an info mapping.
Each iteration sends the observation to a planner and passes the planner's action to
`env.step(action)`. The environment advances the simulator and returns the next observation,
reward, termination flags, and step information.

There are two deliberately distinct planner protocol families:

- Baseline planners use [`PlannerProtocol`](../robot_sf/baselines/interface.py), whose main
  action method is `step(obs) -> dict`.
- Native local-planner integrations use [`LocalPlannerProtocol`](../robot_sf/planner/protocol.py),
  whose action method is `plan(observation) -> (linear_speed, angular_rate)`.

Adapters may bridge these families, but a bridge must keep missing diagnostics, dependency
availability, and fallback/degraded status explicit. A passing adapter test does not establish planner quality or benchmark evidence.

### 4. Produce an episode record

For a lightweight public-API run, [`robot_sf.api.run_episode()`](../robot_sf/api.py) returns an
[`EpisodeRecord`](../robot_sf/benchmark/types.py) containing the episode identity, scenario,
seed, metrics, algorithm name, horizon, timing, and optional raw metadata.

The benchmark path has a separate, richer record contract. [`robot_sf.benchmark.runner`](../robot_sf/benchmark/runner.py)
provides `run_episode()`, `run_batch()`, and `validate_and_write()`. The runner captures
trajectory/metric data, validates records against the versioned episode schema, and appends valid
records to JSONL. Do not use a lightweight API record as a substitute for a benchmark record or
silently omit required provenance.

### 5. Write evidence or diagnostic artifacts

When a workflow produces a tracked diagnostic or evidence artifact, reuse the shared writers in
[`robot_sf/evidence/writers.py`](../robot_sf/evidence/writers.py). They provide deterministic
JSON/CSV/text output, review markers, checksums, and evidence-tree registration hooks. Keep raw
episodes, videos, checkpoints, and other large generated outputs in the worktree-local output
area unless a separate custody contract promotes them to durable storage.

## Package map

| Concern | Canonical owner | What belongs there |
| --- | --- | --- |
| Public entry points | [`robot_sf/api.py`](../robot_sf/api.py) and [`environment_factory.py`](../robot_sf/gym_env/environment_factory.py) | Scenario loading, environment construction, and ergonomic episode execution. |
| Environment and sensors | [`robot_sf/gym_env/`](../robot_sf/gym_env/) | Gymnasium lifecycle, observations, rewards, recording, rendering, and sensor wiring. |
| Simulation and physics | [`robot_sf/sim/`](../robot_sf/sim/) and [`fast-pysf/`](../fast-pysf/) | Robot/pedestrian state transitions and force backends. Preserve site-specific and legacy contracts. |
| Planner protocols and adapters | [`robot_sf/planner/`](../robot_sf/planner/) and [`robot_sf/baselines/`](../robot_sf/baselines/) | Local planners, baseline planners, protocol adapters, and planner diagnostics. |
| Scenarios and maps | [`configs/scenarios/`](../configs/scenarios/) and [`maps/`](../maps/) | Reusable scenario inputs and authored map assets; validate paths and geometry before runtime use. |
| Benchmark records and metrics | [`robot_sf/benchmark/`](../robot_sf/benchmark/) | Versioned record schemas, metric definitions, campaign runners, and provenance checks. |
| Evidence and publication | [`robot_sf/evidence/`](../robot_sf/evidence/) and [`docs/context/evidence/`](context/evidence/) | Artifact writing, catalog registration, diagnostic reports, and bounded evidence handoff. |
| Training and external assets | [`robot_sf/training/`](../robot_sf/training/) and [`docs/external_data_setup.md`](external_data_setup.md) | Training workflows and license/provenance-safe external-data intake. |
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
| Documentation or example | [`docs/`](./) or [`examples/`](../examples/) | Link/path checks, focused docs or example smoke tests, and strict Sphinx validation when navigation changes. |

For terminology such as Gymnasium, Social Navigation Quality Index (SNQI), and vulnerable road
user (VRU), use the [project glossary](glossary.md). For reproducible research workflows, continue
with the [research and benchmark guide](research-guide.md), then consult the relevant benchmark
or evidence contract before interpreting a result.
