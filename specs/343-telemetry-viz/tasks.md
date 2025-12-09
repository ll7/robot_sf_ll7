# Tasks: Data Visualization alongside Pygame

## Phase 1 – Setup
- [X] T001 Ensure venv active and deps installed (`uv sync --all-extras`); source `.venv/bin/activate`.
- [X] T002 Verify artifact root clean and policy in place (`output/` canonical, run `scripts/tools/check_artifact_root.py`).

## Phase 2 – Foundational
- [X] T003 Add telemetry pane feature flags + config plumbing in `robot_sf/gym_env/environment_factory.py` and related config defaults.
- [X] T004 Wire telemetry JSONL writer/manifest to include frame_idx and status fields (append-only) in `robot_sf/telemetry/manifest_writer.py` and `robot_sf/telemetry/sampler.py`.
- [X] T005 Implement off-screen chart rendering helper (matplotlib/agg) producing blit-ready surfaces in `robot_sf/telemetry/visualization.py` (new) or `robot_sf/render/` as appropriate.

## Phase 3 – User Story 1 (P1) Live Pygame + Telemetry panel
- [X] T006 [US1] Integrate docked pane blitting into SDL/Pygame render loop in `robot_sf/render/` (single window, vertical/horizontal split).
- [X] T007 [US1] Expose live panel toggle/metrics selection API via factories; ensure defaults off unless requested (`robot_sf/gym_env/environment_factory.py`).
- [X] T008 [US1] Add health indicators (drop/delay warnings) surfaced via Loguru and run manifest (`robot_sf/telemetry/progress.py` or new helper).
- [X] T009 [US1] Live-mode telemetry JSONL write path under `output/telemetry/<run_id>/telemetry.jsonl` with decimation safeguards.
- [X] T010 [US1] Example/demo script showing live pane (`examples/advanced/22_telemetry_pane.py`) aligned with quickstart.
- [X] T011 [US1] Tests/smokes: live pane rendering + FPS overhead check; assert charts update ≥1 Hz and FPS drop ≤10% (`tests/telemetry/test_live_pane.py`).

## Phase 4 – User Story 2 (P2) Replay with synchronized charts
- [X] T012 [US2] Implement replay loader to sync frames + telemetry (`robot_sf/telemetry/history.py` or new `visualization.py`).
- [X] T013 [US2] Scrubbing controls maintain cursor/frame alignment within one frame (`robot_sf/telemetry/visualization.py`).
- [X] T014 [US2] Export screenshot/clip capturing Pygame view + chart viewport with stable naming under `output/telemetry/<run_id>/`.
- [X] T015 [US2] Tests: replay alignment assertion over 5-minute sample; export artifact existence/content (`tests/telemetry/test_replay.py`).

## Phase 5 – User Story 3 (P3) Headless/CI telemetry sanity
- [X] T016 [US3] Headless run path producing telemetry JSONL + summary PNG/JSON without SDL window (respect `SDL_VIDEODRIVER=dummy`) in `robot_sf/telemetry/visualization.py` or helper.
- [X] T017 [US3] CI artifact guard integration to check non-empty outputs (`scripts/validation` or `tests/telemetry/test_headless.py`).
- [X] T018 [US3] Add headless smoke script/example entry in `examples/` matching quickstart snippet.

## Phase 6 – Polish & Cross-Cutting
- [X] T019 Update docs: link new guide/demo in `docs/README.md` and add usage notes in relevant doc (e.g., `docs/` new page).
- [X] T020 Ensure logging uses Loguru; remove/avoid prints in library code touched.
- [X] T021 Add CHANGELOG.md entry if user-facing behavior changes.
- [X] T022 Validate artifact paths with `scripts/tools/check_artifact_root.py` post-implementation; ensure outputs under `output/` only.
- [ ] T023 Run quality gates: `uv run ruff check --fix . && uv run ruff format .`, `uv run pytest tests`, `uvx ty check . --exit-zero`.
- [ ] T024 Prepare PR notes summarizing behaviors, tests run, and artifacts generated.

## Dependencies
- Story order: US1 (live) → US2 (replay) → US3 (headless) to reuse telemetry pipeline.

## Parallel Opportunities
- [P] T005 (chart helper) can proceed in parallel with T004 (telemetry writer) since distinct modules.
- [P] T010 (demo) can progress while T008 health indicators finalize, once pane API stable.
- [P] T014 (export) can proceed alongside T012/T013 once replay loader skeleton exists.

## Implementation Strategy
- MVP = Phase 3 (US1) with live pane + JSONL + health indicators + FPS budget test.
- Then add replay/export (US2), then headless artifacts (US3), followed by polish.
