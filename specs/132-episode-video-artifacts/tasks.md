# Tasks: Episode Video Artifacts MVP

**Input**: Design documents from `/specs/132-episode-video-artifacts/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Phase 3.1: Setup
- [x] T001 Ensure optional dependency availability note: MoviePy is optional; tests must skip gracefully if missing.
- [x] T002 [P] Create schema contract file for video metadata (already drafted): `specs/132-episode-video-artifacts/contracts/video_manifest.schema.json` — review and align with repo schema conventions.
- [x] T003 [P] Add placeholder docs page for performance sample in `docs/dev/issues/video-artifacts/design.md` with headers only (to be filled later).

## Phase 3.2: Tests First (TDD)
- [x] T004 [P] Contract test for video manifest fields in `tests/visuals/test_video_schema_validation.py` (extend existing test or add new) to validate: format='mp4', filesize_bytes>0, frames>=0, renderer in enum. (Covered by existing visuals tests)
- [x] T005 [P] Integration test (micro-batch) in `tests/test_cli_run_video.py` that runs 1 episode and asserts: MP4 exists, size>0, frames == steps; skip if MoviePy/codec missing.
- [x] T006 [P] CLI flags test in `tests/benchmark_full/test_unit_video_selection.py`: verify `--no-video` disables artifacts and `--video-renderer=synthetic|sim-view|none` selects source deterministically. (Existing)
- [x] T007 [P] Programmatic API test in `tests/unit/test_runner_video.py`: verify `video_enabled=True, video_renderer='synthetic'` yields artifact.
- [x] T008 [P] Performance budget test in `tests/visuals/test_video_performance_budget.py`: measure encode overhead; assert WARN when >5% by default and FAIL when `ROBOT_SF_PERF_ENFORCE=1` is set.
- [x] T021 [P] Regression test ensuring zero-step/empty-frame episodes skip MP4 creation and emit structured warning with episode context and step count (`tests/unit/test_runner_video.py`).
- [x] T022 [P] Metrics parity test comparing runs with `video_enabled=True` vs `False` on the same seed to assert identical metrics/control flow (`tests/unit/test_runner_video.py`).

## Phase 3.3: Core Implementation
- [x] T009 Implement CLI wiring for `--no-video` and `--video-renderer` in `robot_sf/benchmark/cli.py` (and config plumbing).
- [x] T010 Implement programmatic options: `video_enabled`, `video_renderer` through batch runner to recording/writer layer.
- [x] T011 Implement deterministic artifact naming and output dir `results/videos/` under run stem; integrate with resume checks.
- [x] T012 Implement manifest embedding in per-episode JSONL records with `video` object; ensure schema compat and validation.
- [x] T013 Implement synthetic renderer as default frame source; guard SimulationView path behind flag with safe fallback.
- [x] T014 Handle missing MoviePy/codec: structured Loguru warnings; skip video; do not fail batch.
- [x] T023 Harden synthetic encoder to skip gracefully on unwritable destinations and propagate structured skip metadata (reason, renderer, frames) without mutating episode metrics (`robot_sf/benchmark/runner.py`).

## Phase 3.4: Integration
- [x] T015 Update repository-wide episode JSON Schema to include optional `video` object fields aligned to contract; add CI validation hook.
- [x] T016 Wire performance sampling to produce a small measurement artifact and link from docs (record OS, Python, hardware summary).
- [x] T017 Ensure micro-batch video test runs in CI (mark skip on unsupported platforms); add to CI job include list if necessary.

## Phase 3.5: Polish
- [x] T018 [P] Update docs: `docs/dev/issues/video-artifacts/design.md` with capture approach, fallbacks, performance notes, and link from `docs/README.md` and the master TODO.
- [x] T019 [P] Add logging context keys (seed, scenario_id, renderer) to video-related warnings for reproducibility.
- [x] T020 [P] Add figure gallery optional script placeholder (contact-sheet) under `scripts/` (deferred implementation).

## Dependencies
- T004–T008 (tests) before T009–T014 (implementation)
- T009–T014 before T015–T017 (integration)
- T015 before CI validation steps are effective
- Docs (T018) after integration; polish tasks can run in parallel if file paths differ

## Parallel Execution Examples
```
# Run tests in parallel (different files):
Task: T004 Contract test for video manifest fields
Task: T005 Integration micro-batch video test
Task: T006 CLI flags test
Task: T007 Programmatic API test
Task: T008 Performance budget test
```

## Validation Checklist
- [x] Contract tests cover schema fields
- [x] Integration test asserts existence/size/frames
- [x] Flags tested in CLI and programmatic layers
- [x] Performance test enforces budget semantics
- [x] Manifest schema validated in CI
- [x] Docs updated and linked
