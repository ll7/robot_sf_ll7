# Data Model: Environment Factory Ergonomics

**Scope**: Option objects, deprecation mapping, validation outcomes.

## Entities

### RenderOptions
| Field | Type | Default | Description | Validation |
|-------|------|---------|-------------|------------|
| enable_overlay | bool | False | Turn on overlay (debug metrics) | N/A |
| max_fps_override | int | None | Cap rendering FPS if set | >0 |
| ped_velocity_scale | float | 1.0 | Scale factor for ped velocity vectors in visualization | >0 |
| headless_ok | bool | True | Allow creation without display | If False and no display → warn |

### RecordingOptions
| Field | Type | Default | Description | Validation |
|-------|------|---------|-------------|------------|
| record | bool | False | Master switch; if True engage frame capture | N/A |
| video_path | str | None | Output MP4 path; if None buffer only | If provided must end with .mp4 |
| max_frames | int | None | Override global cap (ties to env var) | >0 |
| codec | str | "libx264" | Preferred encoding codec | Non-empty |
| bitrate | str | None | Optional bitrate hint | If set matches pattern | 

### DeprecationMap (internal)
| Legacy Name | New Target | Notes |
|-------------|-----------|-------|
| record_video | recording_options.record | Direct boolean mapping |
| video_output_path | recording_options.video_path | Rename clarity |
| fps | render_options.max_fps_override | Narrow scope |

## Relationships
Environment Factory -> (RenderOptions?, RecordingOptions?) -> Normalized Parameter Set -> Env Instance.

## State Transitions (High-Level)
1. Input Merge: Primitive convenience params + option objects.
2. Normalization: If `record_video=True` and `RecordingOptions.record is False` → set to True, warn about precedence.
3. Validation: Incompatible detection (e.g., record & headless conflict) → auto-adjust or warn.
4. Deprecation: Legacy kw present → map & log warning.

## Validation Rules Summary
- Unknown legacy kw (strict mode) → raise ValueError with suggestion list.
- Unknown legacy kw (permissive mode) → warning + ignore if unmappable.
- `max_frames` if set and exceeds internal hard limit (if any future) → clamp + warning.

## Open Questions (from research)
Referenced in `research.md` Section 4; resolved decisions will update this file accordingly.
