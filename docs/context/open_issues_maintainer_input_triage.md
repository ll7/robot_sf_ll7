# Open Issues Maintainer Input Triage

Date: 2026-05-12

Purpose: track open issues where maintainer input is needed before implementation, and record decisions one question at a time.

## Workflow

- Ask one maintainer question at a time.
- Discuss until the answer is clear.
- Record the decision here.
- Move to the next question only after common understanding is reached.

## Current Priority Questions

### 1. Legacy environment lifecycle

Issues:
- #1150 `refactor: simplify env surface by deprecating or removing legacy env modules`
- #1141 `bug: complete or deprecate SimpleRobotEnv stub implementation`
- #1146 `chore: replace mutable default env configuration in SimpleRobotEnv`
- #1148 `refactor: unify env constructor config defaults and reduce duplicated setup plumbing`

Question:
Should `SimpleRobotEnv` / `EmptyRobotEnv` be removed, kept with deprecation warnings, or converted to aliases/wrappers?

Initial recommendation:
Deprecate first, then remove later if no external callers surface. Prefer folding #1141/#1146 into #1150 rather than implementing another active env path.

Decision:
Pending.

### 2. SocNavBench ETH source assets

Issue:
- #1134 `bench: convert SocNavBench ETH map after official asset staging`

Question:
Can the official SocNavBench ETH assets be staged locally, with source path/checksums/license-safe provenance, or should the issue remain blocked?

Decision:
Pending.

### 3. SDD curation target

Issue:
- #1126 `bench: curate first real SDD-derived benchmark scenario set`

Question:
Which real Stanford Drone Dataset scene/video should be curated first, and can the dataset be staged under the repository artifact policy?

Initial recommendation:
Pick one small canonical scene first and mark it exploratory until validated.

Decision:
Pending.

### 4. Docker-capable runner

Issue:
- #1119 `ci: validate Docker benchmark reproduction smoke on Docker-capable runner`

Question:
Which Docker-capable host or CI runner should run the benchmark reproduction smoke?

Decision:
Pending.

### 5. CARLA-capable host and version

Issue:
- #1111 `feat: run live CARLA T1 oracle replay smoke on a CARLA-capable host`

Question:
Is a CARLA-capable host available, and which CARLA version should be targeted?

Decision:
Pending.

### 6. Dynamic pedestrian occlusion semantics

Issue:
- #1124 `bench: add dynamic pedestrian occlusion filtering for planner observations`

Question:
What dynamic occlusion semantics should v1 use?

Initial recommendation:
Opt-in, disabled by default, circular pedestrian body model, nearest pedestrian blocks line-of-sight, binary visibility only for v1.

Decision:
Pending.

### 7. SensorFusion temporal ordering

Issue:
- #1143 `refactor: define SensorFusion temporal stacking contract for conv extractors`

Question:
Should stacked sensor tensors be ordered oldest-to-newest or newest-to-oldest?

Initial recommendation:
Oldest-to-newest, current frame at `[-1]`, because current code mostly follows that pattern.

Decision:
Pending.

### 8. Multi-AMV first planner adapter

Issue:
- #1128 `bench: integrate multi-AMV runs into primary benchmark outputs`

Question:
Which non-trivial planner family should be the first multi-AMV adapter?

Initial recommendation:
Start with the simplest existing non-RL planner path before expanding.

Decision:
Pending.

### 9. BC warm-start PPO execution environment

Issue:
- #1108 `research: run issue-749 BC warm-start PPO experiment`

Question:
Which execution environment should run the long training job, and where should durable artifacts be promoted?

Decision:
Pending.

### 10. CARLA bridge packaging strategy

Issue:
- #872 `feat: CARLA oracle replay bridge`

Question:
Should the CARLA bridge live as an optional in-repo package or a separate package?

Initial recommendation:
Optional in-repo package guarded by dependency checks.

Decision:
Pending.

## Issues That Do Not Currently Need Maintainer Input

- #1149 `refactor: de-duplicate sensor history stack logic between fusion classes`
- #1147 `bug: make classic_benchmark_full entrypoint fail-closed with actionable behavior`
- #1145 `technical-debt: reduce TODO docstring placeholders in public modules`
- #1144 `docs: clarify WheelSpeedState units in differential drive model`
- #1142 `bug: align visualization video FPS with effective render cadence`
- #1138 `feat: add deterministic obstacle features for predictive planner baseline`
- #1110 `bench: compare CARLA oracle replay metrics against Robot-SF traces`

## Decision Log

- 2026-05-12: Maintainer requested one-question-at-a-time workflow. Created this triage note to preserve the assessment and record decisions.

## Decision Update: Legacy environment lifecycle

Date: 2026-05-12

Maintainer clarification:
- `make_pedestrian_env` is a trainable interface where a robot policy can be loaded and a pedestrian policy can be trained/evaluated against it.
- The desired `EmptyRobotEnv`-style capability is different: a simple scene runner with only Social Force pedestrians simulated, useful even without a trainable ego-pedestrian interface.
- The `EmptyRobotEnv` name is unclear; renaming around crowd/social-force simulation makes sense.

Current assessment:
- `SimpleRobotEnv` appears to be an unfinished Gymnasium/SB3-native sketch. It defines placeholder discrete spaces and no-op `step`, `render`, and `close`, and no references outside refactoring docs/code search were found.
- `EmptyRobotEnv` should not be removed until the useful social-force-only pedestrian simulation capability is preserved or replaced by a clearly named supported factory path.

Updated recommendation:
- Remove `SimpleRobotEnv`; maintainer discussion classified it as an abandoned prototype.
- Replace or rename `EmptyRobotEnv` into a clearer supported surface such as `make_social_force_crowd_env` / `make_crowd_sim_env`, preserving simple Social Force pedestrian simulation.
- Update #1150/#1141 to reflect this distinction before implementation.

## Decision Update: Crowd simulation factory naming

Date: 2026-05-12

Decision:
Use `make_crowd_sim_env` as the supported factory entrypoint name for the preserved Social Force pedestrian/crowd simulation capability.

Implication:
- `EmptyRobotEnv` is replaced by this clearer factory surface.
- `SimpleRobotEnv` has been removed from the supported target surface.

## Decision Update: Crowd simulation API contract

Date: 2026-05-12

Decision:
`make_crowd_sim_env` should expose a programmatic `step`/`reset` interface, not only playback/visualization.

Functional intent:
- Simulate Social Force pedestrians without requiring a robot or trainable ego-pedestrian policy.
- Support data recording of pedestrian behavior in robot-free scenes.
- Steps should advance automatically without a meaningful external action.
- Headless stepping should be as fast as possible; visualization/recording should be optional and opt-in.

Implication:
- Prefer a Gymnasium-compatible API with an action space that is empty/no-op or otherwise clearly indicates autonomous stepping.
- Keep visualization separate from the fast path.

## Decision Update: Crowd simulation step action handling

Date: 2026-05-12

Decision:
`make_crowd_sim_env` should support `step(action=None)`.

Contract:
- Callers may use `env.step()` for autonomous stepping.
- Callers may also use Gymnasium-style `env.step(action)`.
- Since crowd simulation has no external control input, non-`None` actions should be ignored with a warning rather than rejected.
- The warning should make it clear that pedestrians advance autonomously and action input has no effect.

## Decision Update: EmptyRobotEnv implementation strategy

Date: 2026-05-12

Evidence:
A repository search for `EmptyRobotEnv` / `empty_robot_env` found no active code or test imports. References are limited to the legacy file itself and historical/refactoring docs.

Decision:
Do not reuse `EmptyRobotEnv` as the implementation base for `make_crowd_sim_env`.

Implementation direction:
- Remove `EmptyRobotEnv` as legacy/unreferenced code once `make_crowd_sim_env` exists.
- Build `make_crowd_sim_env` as a clean implementation for the explicitly discussed Social Force crowd-simulation use case.
- Update historical/refactoring docs only as needed to avoid stale references.

## Decision Update: Crowd simulation config shape

Date: 2026-05-12

Decision:
Add a narrow `CrowdSimulationConfig` for `make_crowd_sim_env`.

Contract:
- Reuse existing primitives such as `SimulationSettings`, `MapDefinitionPool`, map selection, telemetry/recording conventions, and seed/max-step style controls.
- Do not inherit from robot-centered `EnvSettings` / `RobotSimulationConfig`.
- Do not inherit from `PedestrianSimulationConfig`, because that config is for trainable ego-pedestrian-vs-robot workflows.
- Avoid exposing robot, lidar, planner, image-observation, or predictive-foresight fields unless the crowd-only implementation actually needs them.

Rationale:
A narrow config avoids misleading users into thinking robot/planner/sensor knobs affect robot-free crowd simulation, while still avoiding a fully separate config ecosystem.

## Decision Update: Crowd simulation observation shape

Date: 2026-05-12

Decision:
`make_crowd_sim_env.reset()` and `make_crowd_sim_env.step()` should return a compact structured observation dict, not the benchmark episode schema.

Initial observation direction:
- Include pedestrian positions.
- Include pedestrian velocities when available.
- Include pedestrian goals/routes when available.
- Keep benchmark episode export/recording as a separate adapter or optional recording layer.

Rationale:
The environment API should be lightweight and useful for direct programmatic simulation. Benchmark schemas can be produced from recorded states without forcing the env observation contract to mirror benchmark outputs.

## Decision Update: Crowd simulation static metadata

Date: 2026-05-12

Decision:
Keep fast stepping lean. Do not repeat static map geometry/obstacles in every `step()` observation.

Contract:
- `step()` returns dynamic pedestrian state only, plus minimal status fields needed for control flow.
- Static scene metadata such as map geometry, obstacles, spawn/goal zones, and map id should be available from `reset()` info, env metadata, or a dedicated accessor.
- Recording/export layers may combine static metadata with dynamic states when producing artifacts.

## Decision Update: Crowd simulation recording scope

Date: 2026-05-12

Decision:
Include optional JSONL-style recording in v1 of `make_crowd_sim_env`.

Contract:
- Recording should be opt-in.
- Default headless stepping remains as fast as possible.
- Recorded output should support pedestrian-only behavior analysis and later benchmark/artifact adapters.
- Static metadata should be written once per run/manifest or reset record, not repeated in every step unless explicitly needed.

## Decision Update: Crowd simulation package location

Date: 2026-05-12

Decision:
Place the v1 crowd simulation env implementation under `robot_sf.gym_env`.

Contract:
- Add implementation module such as `robot_sf/gym_env/crowd_sim_env.py`.
- Expose user-facing constructor as `make_crowd_sim_env` from `robot_sf/gym_env/environment_factory.py`.
- Keep lower-level reusable simulator primitives in `robot_sf.sim` only if/when they naturally emerge.

Rationale:
The feature has env-style `step`/`reset`, observations, info, optional recording, and optional rendering. It is user-facing environment API, not only backend simulation glue.

## Decision Update: Crowd simulation rendering scope

Date: 2026-05-12

Decision:
`make_crowd_sim_env` should support rendering in v1.

Contract:
- Rendering should be optional and opt-in.
- Default headless stepping remains as fast as possible.
- Rendering should visualize Social Force pedestrian-only scenes without requiring a robot.
- Rendering support should not force static map geometry into every step observation.
- JSONL recording and rendering should be independently configurable.

## Decision Update: Crowd simulation rendering and implementation surface

Date: 2026-05-12

Decision:
`make_crowd_sim_env` should support rendering in v1 while keeping the default path headless and fast.

Implementation direction:
- Add a dedicated `CrowdSimEnv` rather than routing through `RobotEnv`.
- Use `Simulator` with an empty robot list so Social Force pedestrians advance automatically.
- Keep `step(action=None)` compatible with Gymnasium; non-`None` actions are ignored with a warning.
- Return compact dynamic pedestrian observations: positions, velocities, current goals, latest forces, and step index.
- Keep static scene metadata in `reset`/`step` info and optional recording metadata, not in every observation payload.
- Lazily create `SimulationView` only for `render_mode` or video capture.
- Pass a robot-free visual state to the renderer so rendering does not invent a fake robot.
- Include optional compact JSONL recording for reset/step events.
