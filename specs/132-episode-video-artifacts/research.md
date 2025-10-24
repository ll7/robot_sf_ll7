# Research: Episode Video Artifacts MVP

Date: 2025-09-24

## Decisions
- Flags: `--no-video` toggle; `--video-renderer=synthetic|sim-view|none`.
- Programmatic API: `video_enabled: bool`, `video_renderer: Literal['synthetic','sim-view','none']`.
- Defaults: Video disabled by default; default renderer `synthetic`.
- Output: `results/videos/<scenario>__<seed>__<episode_idx>.mp4`; overwrite with resume-aware checks.
- Manifest: In-episode JSONL `video` object: `{ path, format: 'mp4', filesize_bytes, frames, renderer, notes? }`.
- Performance: Overhead <5% target; WARN by default; FAIL only when `ROBOT_SF_PERF_ENFORCE=1`.
- Resilience: If MoviePy/codec missing, skip video with structured warning.
- Test Scope: Micro-batch (1–2 episodes) asserting existence, size>0, frames==steps.

## Rationale
- Synthetic renderer as default avoids display/codec pitfalls on CI and macOS while still producing informative artifacts.
- In-episode manifest metadata aligns with existing JSONL patterns, simplifying aggregation and resume.
- Soft enforcement for performance avoids flaky CI while enabling strict local checks.

## Alternatives Considered
- Sidecar manifest for videos — rejected due to added complexity and sync risks.
- Auto renderer selection — rejected; explicit `sim-view` reduces hidden dependencies.
- Hard fail on performance overage — rejected; too brittle for CI variance.
