# Migration Guide: Environment Factory Ergonomics (Feature 130)

Status: In Progress (pre-release)

## Overview
This guide explains the ergonomic improvements to environment factory functions (`make_robot_env`, `make_image_robot_env`, `make_pedestrian_env`) and how to migrate from legacy keyword arguments to the new structured option objects while preserving backward compatibility.

## Goals
- Improve discoverability and IDE autocomplete with explicit parameters.
- Introduce structured `RenderOptions` and `RecordingOptions` dataclasses.
- Provide a deprecation shim for legacy kwargs with clear Loguru warnings.
- Maintain performance (< +10% creation mean, target +5%).

## Deprecation Window
| Phase | Action | Timeline |
|-------|--------|----------|
| Initial Release | Warnings for legacy kwargs (strict error on unknown unless permissive mode) | Current version |
| +1 Minor | Continue warnings; update docs to remove legacy examples | Next minor |
| +2 Minor | Potential removal of mapping layer (subject to adoption metrics) | Two minors ahead |

## Legacy to New Mapping
| Legacy Kwarg | New Form | Notes |
|--------------|----------|-------|
| `record_video` | `RecordingOptions(record=True)` OR keep boolean convenience | Boolean convenience retained |
| `video_output_path` | `RecordingOptions(video_path=...)` | Renamed for clarity |
| `fps` | `RenderOptions(max_fps_override=...)` | Scopes rendering concerns |

## Example Migration
**Before**
```python
env = make_robot_env(debug=True, record_video=True, video_path="episode.mp4", fps=30)
```
**After (Structured)**
```python
from robot_sf.gym_env.environment_factory import make_robot_env, RecordingOptions, RenderOptions

render_opts = RenderOptions(max_fps_override=30)
rec_opts = RecordingOptions(record=True, video_path="episode.mp4")
env = make_robot_env(debug=True, render_options=render_opts, recording_options=rec_opts)
```
**After (Mixed Convenience)**
```python
env = make_robot_env(record_video=True, video_path="episode.mp4")
```

## Precedence Rules
| Scenario | Outcome | Log |
|----------|---------|-----|
| `record_video=True` + `RecordingOptions(record=False)` | Recording enabled (flip) | WARNING with precedence message |
| `video_fps=30` + `RenderOptions(max_fps_override=20)` | Uses 20 | INFO (creation) |
| Legacy kw + strict mode | Error | ValueError |
| Legacy kw + permissive env (`ROBOT_SF_FACTORY_LEGACY=1`) | Warning + mapping | WARNING |

## Logging Diagnostics
- INFO: Creation line (env type, record flags, fps, video path).
- WARNING: Each mapped legacy kw.
- WARNING: Precedence override when boolean convenience conflicts with explicit options.

## Performance Considerations
Baseline results stored in `results/factory_perf_baseline.json`. Regression guard test ensures mean creation time stays within +10% (target +5%). Use `scripts/perf/baseline_factory_creation.py` to regenerate baselines after significant optimization.

## Seed Determinism
A relaxed test (`test_seed_determinism.py`) attempts to compare a stable slice of the initial observation. Full determinism is not enforced yet if stochastic components are present; test will skip in that case.

## Recording Path
Using `record_video=True` or an explicit `RecordingOptions(record=True)` triggers creation of a `SimulationView` (even if `debug=False`) to buffer frames. Video encoding still depends on downstream flush logic (unchanged by this feature).

## Environment Variable Controls
| Variable | Effect |
|----------|--------|
| `ROBOT_SF_FACTORY_LEGACY=1` | Permissive legacy mode (unknown legacy kwargs ignored w/ warning) |
| `ROBOT_SF_FAST_DEMO=1` | Fast path for classic interactions demo (test performance aid; not part of public API) |

## Migration Strategy Recommendations
1. Replace `fps` with a `RenderOptions` instance in new code.
2. Gradually phase out `video_output_path` in favor of `RecordingOptions(video_path=...)`.
3. Keep `record_video` convenience for quick prototypes; prefer explicit dataclass in production scripts.
4. Monitor logs for warnings during transition.

## Future Work (Deferred)
- Multi-robot ergonomic factory (`_make_multi_robot_env_future`).
- Potential telemetry hook (stub tasks T027/T028).
- Stronger determinism guarantees for initial observations.

## References
- Spec: `specs/130-improve-environment-factory/spec.md`
- Tasks: `specs/130-improve-environment-factory/tasks.md`
- Data Model: `specs/130-improve-environment-factory/data-model.md`
- Quickstart: `specs/130-improve-environment-factory/quickstart.md`

---
*Document generated as part of Task T018.*
