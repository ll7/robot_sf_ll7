# Episode Video Artifacts – Design

Purpose: capture the minimal viable workflow for recording per-episode MP4
artifacts during benchmark runs while keeping CI and resume flows stable.

## Goals & Non-Goals
- Provide deterministic MP4 output per episode when explicitly requested.
- Keep benchmarks reproducible: artifacts are optional and resume-aware.
- Maintain soft (<5%) encoding overhead with opt-in hard enforcement.
- Support headless CI environments (synthetic renderer default).
- `sim-view` renderer is intentionally postponed until the visual stack is
  hardened; only the synthetic renderer is wired today.

## Renderer Modes & Flags
- CLI: `--video-renderer` accepts `synthetic`, `sim-view`, or `none`
  (`--no-video` overrides the renderer entirely).
- Programmatic API: `run_batch(..., video_enabled=True, video_renderer="synthetic")`.
- Default behaviour keeps video disabled to avoid unexpected runtime costs.
- When MoviePy/ffmpeg is unavailable the runner emits a structured warning with
  `episode_id`, `scenario_id`, `seed`, and `renderer` to surface context.

## Output Layout & Naming
- Videos live under `results/videos/` next to the JSONL stem. Filenames follow
  `video_<episode_id>.mp4` for resume-friendly deterministic lookups.
- Failures leave JSONL intact; the runner logs why the artifact was skipped.

## Manifest Schema
- Each episode record now carries an optional `video` object with:
  `path`, `format='mp4'`, `filesize_bytes`, `frames`, `renderer`, optional
  `notes`, plus performance fields `encode_seconds` and `overhead_ratio`.
- Schema updates are mirrored in:
  - `robot_sf/benchmark/schemas/episode.schema.v1.json`
  - `robot_sf/benchmark/schemas/episode.schema.v1.json`
  - Contract copy under `specs/120-social-navigation-benchmark-plan/contracts/`.
- Contract tests validate the schema and the integration test asserts presence
  of the manifest when videos are produced.

## Performance Budget & Telemetry
- Environment variables control soft/hard thresholds:
  - `ROBOT_SF_VIDEO_OVERHEAD_SOFT` (default `0.10`)
  - `ROBOT_SF_VIDEO_OVERHEAD_HARD` (default `0.50`)
  - `ROBOT_SF_PERF_ENFORCE=1` escalates soft/hard breaches into errors.
- Warnings include `{episode_id, scenario_id, seed, renderer}` for traceability.
- Optional snapshot: set `ROBOT_SF_VIDEO_PERF_SNAPSHOT=1` to write
  `videos/perf_snapshot.json` containing encode totals, ms/frame, and platform
  info.

## CI & Headless Support
- GitHub Actions installs ffmpeg and relies on the synthetic renderer, so the
  micro-batch video integration test runs as part of the default `pytest` job.
- All video tests skip gracefully when MoviePy or Loguru is missing to keep
  developers unblocked on minimal environments.

## Future Work
- Integrate SimulationView (`sim-view`) sampling once headless GL issues are
  resolved.
- Implement `scripts/generate_video_contact_sheet.py` once renderer fidelity and
  performance budgets are finalised.
