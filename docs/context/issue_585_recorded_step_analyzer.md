# Issue 585 Recorded-Step Analyzer

Related issue: `#585`

## Goal

Add a recorded-scenario playback analyzer that keeps the rendered frame aligned with the reward
decomposition and step-level metrics that produced that frame's score.

## Key Decision

Keep recorded state playback and telemetry replay as separate artifact streams, then link them
through additive episode metadata:

- telemetry JSONL keeps per-step analyzer payloads (`reward_terms`, `reward_total`,
  `step_metrics`, `episode_id`) without changing the live-pane contract,
- JSONL episode sidecars record `telemetry_path` plus `telemetry_episode_id`,
- playback loads those telemetry samples lazily and offsets frame indices when multiple episodes are
  combined into one interactive session.

This keeps older recordings readable and avoids retrofitting analysis-only fields into
`VisualizableSimState`.

## Implementation Surface

- `robot_sf/gym_env/robot_env.py`
  - emit episode-scoped telemetry payloads with reward-term and selected step-metric fields.
- `robot_sf/render/jsonl_recording.py`
  - persist telemetry linkage in episode metadata.
- `robot_sf/render/jsonl_playback.py`
  - load telemetry samples back into `PlaybackEpisode` when metadata provides a link.
- `robot_sf/telemetry/history.py`
  - normalize raw telemetry samples into a structured replay snapshot for analyzer consumers.
- `robot_sf/render/interactive_playback.py`
  - render the analyzer overlay and support visible-metric filtering.

## Validation

Focused contract checks:

```bash
uv run pytest tests/telemetry/test_replay.py tests/test_jsonl_recording.py tests/test_interactive_playback_enhanced.py
uv run ruff check robot_sf/telemetry/history.py robot_sf/render/jsonl_recording.py robot_sf/gym_env/base_env.py robot_sf/gym_env/robot_env.py robot_sf/render/jsonl_playback.py robot_sf/render/interactive_playback.py tests/telemetry/test_replay.py tests/test_jsonl_recording.py tests/test_interactive_playback_enhanced.py
```

Broad readiness gate after latest-main sync:

```bash
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

## Remaining Risk

The analyzer assumes telemetry recording was enabled for the original run. Older or minimal
recordings remain playable but will not show analyzer content because there is no telemetry stream
to align against.
