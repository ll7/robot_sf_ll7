"""Data Model: Classic Interactions PPO Visualization (Feature 128)

Describes the structured episode summary returned by `run_demo`.
"""

# EpisodeSummary Schema

| Field      | Type   | Description |
|------------|--------|-------------|
| scenario   | str    | Scenario name from classic interactions matrix |
| seed       | int    | Episode seed (deterministic ordering as listed) |
| steps      | int    | Number of environment steps executed |
| outcome    | str    | One of {success, collision, timeout, done} |
| success    | bool   | True if environment reported success termination |
| collision  | bool   | True if collision flagged in final info dict |
| timeout    | bool   | True if timeout flagged in final info dict |
| recorded   | bool   | True if an MP4 was successfully written for this episode |

## Invariants

1. Exactly one of `success`, `collision`, `timeout` may be True (else all False → outcome `done`).
2. `steps >= 0` and counts full steps including terminal transition.
3. `recorded` is True only when recording enabled AND moviepy/ffmpeg available AND encoding succeeded.
4. `outcome` mirrors the boolean flags with precedence: collision > success > timeout > done.

## Rationale

The split booleans (success/collision/timeout) enable downstream aggregation without string parsing.
`recorded` provides a quick filter for artifact presence. A future extension may add `video_path`.

## Extension Hooks (Deferred)

- Multi-scenario chaining (FR-022) would add a `scenario_index` field.
- Frame sampling (FR-023) may add `frame_stride` metadata.

## Example

```json
{
  "scenario": "classic_crossing_low",
  "seed": 101,
  "steps": 342,
  "outcome": "success",
  "success": true,
  "collision": false,
  "timeout": false,
  "recorded": false
}
```
