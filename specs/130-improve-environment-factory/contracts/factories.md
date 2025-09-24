# Factory Contracts

Defines the public parameter contracts and validation behaviors for environment factory functions after ergonomics improvements.

## make_robot_env
| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| config | RobotSimulationConfig | No | new RobotSimulationConfig() | Unified simulation configuration |
| debug | bool | No | False | Enable interactive / detailed rendering mode |
| seed | int | No | None | Deterministic seed for env RNG |
| record_video | bool | No | False | Convenience flag to enable recording (creates RecordingOptions if none) |
| video_path | str | No | None | Convenience path (paired with record_video) |
| reward_func | Callable | No | None | Optional override reward function |
| max_episode_steps | int | No | None | Optional wrapper to cap episode length |
| render_options | RenderOptions | No | None | Advanced rendering customization |
| recording_options | RecordingOptions | No | None | Advanced recording customization |

Validation Rules:
- If both `record_video` and `recording_options.record` provided: `recording_options` wins; emit WARNING.
- If `record_video` True and `video_path` not provided: proceed (buffer-only) and WARN about missing output path.
- If invalid combination (future reserved): raise ValueError with actionable message.

## make_image_robot_env
Same as `make_robot_env` plus image-specific fields inside `config` (no additional params added here to avoid signature expansion).

## make_pedestrian_env
Adds:
| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| robot_model | Any | Yes | N/A | Trained policy/model required for pedestrian environment |

Otherwise inherits core contract rules.

## Deprecation Handling
Legacy kwargs mapping (non-exhaustive initial set):
| Legacy | New | Action |
|--------|-----|--------|
| record_video | recording_options.record | Map + WARN |
| video_output_path | recording_options.video_path | Map + WARN |
| fps | render_options.max_fps_override | Map + WARN |

Unknown legacy kw:
- Strict mode (default): raise ValueError listing allowed params.
- Permissive (`ROBOT_SF_FACTORY_LEGACY=1`): WARN + ignore if unmapped.

## Logging Contract
| Event | Level | Message Pattern |
|-------|-------|-----------------|
| Factory creation | INFO | Creating {factory} debug={debug} record={record} seed={seed} |
| Deprecated kw used | WARNING | Deprecated parameter '{legacy}' mapped to '{new}' | 
| Unknown kw (permissive) | WARNING | Unknown parameter '{legacy}' ignored (enable strict mode) |
| Conflict (boolean vs object) | WARNING | Boolean flag overridden by explicit options object for recording |
| Missing video path | WARNING | Recording enabled but no video_path provided; buffering frames only |

## Failure Modes
- `ValueError` for unknown kw (strict) or invalid value domain.
- `TypeError` if `config` not instance of expected config class.

## Non-Goals
- Introducing asynchronous recording pipeline.
- Changing reward function semantics.
- Altering observation or action space definitions.
