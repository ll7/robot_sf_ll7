# Quickstart: Improved Environment Factory Ergonomics

This guide shows before/after usage patterns for the refactored environment factory API.

## Before (Legacy Style)
```python
from robot_sf.gym_env.environment_factory import make_robot_env

env = make_robot_env(debug=True, record_video=True, video_path="run.mp4", fps=30)
```

Issues:
- Implicit `**kwargs` uncertain.
- No structured grouping for recording vs rendering.
- Potential conflict (recording with debug False) unclear until runtime.

## After (New Ergonomic API)
```python
from robot_sf.gym_env.environment_factory import make_robot_env, RecordingOptions, RenderOptions

render_opts = RenderOptions(max_fps_override=30)
rec_opts = RecordingOptions(record=True, video_path="run.mp4")
env = make_robot_env(debug=True, recording_options=rec_opts, render_options=render_opts, seed=42)
```

### Minimal Usage
```python
env = make_robot_env()
```

### Enable Recording Quickly (Convenience Boolean)
```python
env = make_robot_env(record_video=True, video_path="episode.mp4")
```

Under the hood this creates/normalizes a `RecordingOptions` object. Explicit object overrides boolean if both provided (warning emitted).

### Image Observation Environment
```python
img_env = make_image_robot_env(render_options=RenderOptions(max_fps_override=24))
```

### Pedestrian Environment with Existing Policy
```python
policy = load_trained_policy("model/ppo_model.zip")  # user helper
ped_env = make_pedestrian_env(robot_model=policy, recording_options=RecordingOptions(record=True))
```

## Migration Notes
| Legacy Param | New Form | Action |
|--------------|----------|--------|
| record_video | RecordingOptions.record | Direct mapping |
| video_path / video_output_path | RecordingOptions.video_path | Rename |
| fps | RenderOptions.max_fps_override | Rename + scope |

## Warnings & Diagnostics
- If `record_video=True` and `debug=False` → recording still works (headless) with warning clarifying reduced overlay availability.
- Unknown legacy kw triggers error unless `ROBOT_SF_FACTORY_LEGACY=1` set (then warning + attempt mapping).

## Performance Considerations
- Option object construction negligible (<0.1 ms typical); monitored in performance validation.
- Recording and image observations remain the dominant cost centers; unchanged by ergonomics refactor.

## Next Steps
Refer to `data-model.md` for field definitions and to migration guide (forthcoming) in `docs/dev/issues/130-improve-environment-factory/`.
