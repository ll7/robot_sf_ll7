<!--
Guidance for enabling SocNavBench-style structured observations and planner adapters in Robot SF.
-->

# SocNav Structured Observations & Planner Adapters

## Enable SocNav structured observations
- Set `observation_mode=ObservationMode.SOCNAV_STRUCT` on `RobotSimulationConfig` (or legacy `EnvSettings`).
- Factories: `make_robot_env(config=RobotSimulationConfig(observation_mode=ObservationMode.SOCNAV_STRUCT))`.
- The observation becomes a structured `spaces.Dict` with robot pose/heading/speed/radius, current/next goal, padded pedestrian positions/count, map size, and sim timestep.

## Use planner adapters
- Lightweight adapter: `SamplingPlannerAdapter` consumes the structured observation and emits `(v, w)` directly.
- SocNavBench wrapper: `SocNavBenchSamplingAdapter` resolves the upstream SocNavBench sampling planner (default root `third_party/socnavbench`, override with `ROBOT_SF_SOCNAV_ROOT` or `socnav_root`). By default it requires upstream dependencies; set `allow_fallback=True` to use the lightweight adapter when missing.
- External SocNavBench roots outside the repository require `ROBOT_SF_SOCNAV_ALLOW_UNTRUSTED_ROOT=1` to opt in.
- Policy wrapper: `SocNavPlannerPolicy(adapter)` provides an `.act(obs)` helper for Gym loops.

### Minimal code snippet
```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.socnav import SocNavPlannerPolicy

config = RobotSimulationConfig(observation_mode=ObservationMode.SOCNAV_STRUCT)
env = make_robot_env(config=config)
policy = SocNavPlannerPolicy()

obs, _ = env.reset()
action = policy.act(obs)
obs, reward, terminated, truncated, info = env.step(action)
```

## Example script
- See `examples/advanced/18_socnav_structured_observation.py` for a runnable demo that loops until termination using the planner policy.

## Notes
- Upstream planner integration is explicit; ensure `third_party/socnavbench` (or `ROBOT_SF_SOCNAV_ROOT`) is present if you need the original sampling planner. Use `allow_fallback=True` only for non-benchmark fallback behavior.
- If you use a SocNavBench checkout outside this repository, set `ROBOT_SF_SOCNAV_ALLOW_UNTRUSTED_ROOT=1` to acknowledge the trust boundary.
- Default mode remains `DEFAULT_GYM`; enabling SocNav mode is opt-in to avoid breaking existing agents.
