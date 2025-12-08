# Quickstart: Telemetry Visualization alongside Pygame

## Live Run with Docked Telemetry Pane

```python
from robot_sf.gym_env.environment_factory import make_robot_env

env = make_robot_env(debug=True, enable_telemetry_panel=True, telemetry_metrics=["fps", "reward", "collisions", "min_ped_distance", "action_norm"])
obs, info = env.reset()
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break
env.close()
```
- The Pygame window shows the docked telemetry pane blitted with charts (call `env.render()` to drive the SDL event loop and keep the window open).
- Telemetry JSONL writes to `output/telemetry/<run_id>/telemetry.jsonl` (artifact policy compliant).

## Headless Smoke (CI-friendly)

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
ROBOT_SF_ARTIFACT_ROOT=output \
uv run python - <<'PY'
from robot_sf.gym_env.environment_factory import make_robot_env

env = make_robot_env(debug=False, enable_telemetry_panel=False, telemetry_record=True)
obs, info = env.reset()
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break
env.close()
PY
```
- Produces `telemetry.jsonl` and a summary PNG/JSON in `output/telemetry/<run_id>/` without needing a display.

## Replay and Export

```python
from robot_sf.telemetry.history import load_telemetry_stream
from robot_sf.telemetry.visualization import replay_episode

stream = load_telemetry_stream("output/telemetry/<run_id>/telemetry.jsonl")
replay = replay_episode(stream, frames_path="output/telemetry/<run_id>/frames")
replay.scrub(to_frame=150)
replay.export_clip(start_frame=120, end_frame=180, out_path="output/telemetry/<run_id>/clip.mp4")
```
- Playback keeps chart cursor aligned within one frame of the video.
- Exports capture both the Pygame view and chart viewport for the selected range.

## Validation Checks

- Live: verify FPS drop ≤10% vs baseline scenario with panel off.
- Replay: ensure cursor alignment assertion passes for 5-minute run.
- Headless: artifact guard reports non-empty telemetry JSONL + summary graphic.
