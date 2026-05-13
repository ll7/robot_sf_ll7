# Issue 1151 manual-control MVP foundation

Date: 2026-05-12

## Scope implemented so far

This branch adds the first pure, testable foundations for the Pygame manual-control benchmark recorder epic. It does not yet implement the interactive Pygame runner, renderer integration, baseline loader, or full benchmark comparison loop.

Implemented artifacts:

- `robot_sf/manual_control/input_mapping.py`
  - `ManualKeyState`
  - `DifferentialDriveKeyboardMapper`
  - `manual_keyboard_diff_drive_hold_v1` metadata
- `robot_sf/manual_control/session.py`
  - `ManualSessionController`
  - `ManualSessionState`
  - `AttemptKey` / `AttemptProgress`
- `robot_sf/manual_control/recording.py`
  - `ManualSessionMetadata`
  - `ManualControlRecord`
  - `ManualJsonlRecorder`
- Tests:
  - `tests/test_manual_control_input_mapping.py`
  - `tests/test_manual_control_session.py`
  - `tests/test_manual_control_recording.py`

## Design decisions

### Input mapping

The initial MVP mapper is differential-drive only and uses hold-to-command keyboard semantics. Held keys define target robot velocities; the emitted action is the delta from the current differential-drive velocity to that target.

This matches the current differential-drive action model, where actions are applied as velocity deltas rather than absolute target velocities.

Default key groups:

- Forward: `w`, `up`
- Backward: `s`, `down`
- Left: `a`, `left`
- Right: `d`, `right`
- Brake: `space`, `brake`, `stop`

When no movement key is held, the mapper emits a delta back to zero velocity. Reverse motion is only targeted when `DifferentialDriveSettings.allow_backwards=True`.

### Session lifecycle

The session controller is intentionally pure and renderer-independent. It owns:

- countdown before stepping,
- pause/resume state,
- positive speed multiplier validation,
- terminal attempt bookkeeping,
- retry count,
- completed attempts,
- unresolved scenario/seed attempts where the human did not beat the baseline.

The future Pygame runner should use `controller.should_step` as the gate before calling `env.step(...)`.

### Recording contract

Manual-control JSONL records use `record_schema=manual_control_v1` and include:

- session metadata,
- scenario id,
- seed,
- attempt id,
- step index,
- event type,
- input keys,
- mapped action,
- optional observation payload,
- metrics,
- `training_sample` flag,
- policy-to-beat id/source metadata.

Pause/countdown/retry events can be recorded with `training_sample=false`; behavior-cloning extraction should use only records explicitly marked `training_sample=true` unless a future exporter applies additional filtering.

## Remaining MVP work

- Add an interactive runner under `robot_sf/manual_control/` or `scripts/manual_control/`.
- Wire Pygame events into `ManualKeyState` and `ManualSessionController`.
- Step `RobotEnv` with `DifferentialDriveKeyboardMapper.map_action(...)`.
- Add overlays for goal, current steering/action state, countdown, pause, speed multiplier, episode status, and terminal/failure reason.
- Record simulator state and per-step metrics into `ManualControlRecord` rows.
- Load and freeze policy-to-beat metadata at session start.
- Add baseline comparison logic and session manifest summaries.
- Add a documented smoke command and targeted tests for runner orchestration.

## Validation status

Validation has been run for this implementation slice:

- `UV_NO_CONFIG=1 python -m pytest tests/test_manual_control_*.py -q`
- `UV_NO_CONFIG=1 BASE_REF=origin/main scripts/dev/pr_ready_check.sh`

These checks cover the manual-control foundation helpers, BC export CLI path, and the
branch-wide readiness gate for the main-based PR branch.

## Additional foundation: baseline comparison

Implemented artifacts:

- `robot_sf/manual_control/baseline.py`
  - `MetricDirection`
  - `BaselineMetric`
  - `PolicyBaseline`
  - `BaselineComparison`
- Tests:
  - `tests/test_manual_control_baseline.py`

Design:

- The policy-to-beat is represented as a frozen `PolicyBaseline` with a source string that should identify the config/artifact/registry entry used for the session.
- The baseline has one primary metric that decides `beat_baseline`.
- Metrics declare whether higher or lower values are better.
- Optional tolerance prevents tiny numeric differences from counting as a meaningful win.
- Missing primary candidate metrics fail closed with `KeyError` instead of silently treating the attempt as comparable.

Remaining integration work:

- Load a real `PolicyBaseline` from a config, registry entry, or command-line manifest selected at runner startup.
- Store `PolicyBaseline.to_manifest_dict()` in the session manifest.
- Use `PolicyBaseline.compare(...)` when an attempt reaches terminal state and feed the result into `ManualSessionController.mark_terminal(...)`.

## Additional foundation: demonstration export

Implemented artifacts:

- `robot_sf/manual_control/export.py`
  - `DemonstrationSample`
  - `export_demonstration_samples`
- Tests:
  - `tests/test_manual_control_export.py`

Design:

- Only records with `training_sample=true` are exported into BC samples.
- Non-training events such as pause, countdown, retry, and metadata-only records remain in the JSONL stream but are filtered from BC samples.
- Training-marked records must include both `observation` and `mapped_action`; missing alignment fails closed with `ValueError`.
- Exported samples use `sample_schema=manual_control_bc_v1` and preserve session, scenario, seed, attempt, step, observation, action, and input keys.

## Additional foundation: mapper dispatch

Implemented artifacts:

- `robot_sf/manual_control/input_mapping.py::mapper_for_robot_config`

Design:

- The MVP supports `DifferentialDriveSettings` through `DifferentialDriveKeyboardMapper`.
- Other action spaces intentionally fail closed with `NotImplementedError` until explicit mapper versions are added.
- This gives follow-up steering/view-mode work a single dispatch point for holonomic, bicycle, mouse, cruise, or ego-view-specific mapper variants.

## Additional foundation: JSONL loading

Implemented artifacts:

- `robot_sf/manual_control/recording.py::ManualControlRecord.from_json_dict`
- `robot_sf/manual_control/recording.py::load_manual_jsonl_records`

Design:

- Manual-control JSONL streams can be loaded back into typed `ManualControlRecord` instances.
- Loading fails closed on unsupported `record_schema`, malformed JSON, non-object lines, or missing session metadata.
- This is the read-side prerequisite for future replay and behavior-cloning export commands.

## Additional foundation: JSONL-to-BC export

Implemented artifacts:

- `robot_sf/manual_control/export.py::export_demonstration_samples_from_jsonl`

Design:

- The convenience exporter loads schema-validated manual-control JSONL records and extracts only `training_sample=true` rows.
- This provides the compact BC extraction path required before a CLI/export command is added.

## Additional foundation: compact BC sample writing

Implemented artifacts:

- `robot_sf/manual_control/export.py::write_demonstration_samples_jsonl`

Design:

- Extracted `DemonstrationSample` instances can be written as sorted-key JSONL using `sample_schema=manual_control_bc_v1`.
- This provides the derived dataset write path for future CLI/export commands.

## Additional foundation: replay grouping

Implemented artifacts:

- `robot_sf/manual_control/replay.py::group_records_by_attempt`
- `robot_sf/manual_control/replay.py::ManualAttemptReplay`

Design:

- Manual-control records can be grouped by scenario id, seed, and attempt id.
- Records inside each replay group preserve append-only source-stream order.
- This is a data-ordering prerequisite for future visual replay and rewind work; it does not restore simulator state or render playback by itself.

## Additional foundation: session manifest

Implemented artifacts:

- `robot_sf/manual_control/manifest.py::ManualSessionManifest`
- `robot_sf/manual_control/manifest.py::write_manual_session_manifest`

Design:

- The manifest uses `manifest_schema=manual_control_session_manifest_v1`.
- It records session metadata, optional frozen policy baseline metadata, completed attempts, unresolved attempts, artifact paths, and notes.
- This is the JSON manifest surface the future runner should write at session end.

## Additional foundation: BC export CLI

Implemented artifacts:

- `scripts/manual_control/export_bc_samples.py`
- `tests/test_manual_control_export_cli.py`

Design:

- The CLI reads a manual-control session JSONL through the schema-validating loader.
- It writes compact `manual_control_bc_v1` samples to a destination JSONL.
- Only records marked `training_sample=true` are exported.

Usage sketch:

```bash
uv run python scripts/manual_control/export_bc_samples.py \
  --input output/manual_control/session.jsonl \
  --output output/manual_control/bc_samples.jsonl
```

## Additional foundation: mode identifiers

Implemented artifacts:

- `robot_sf/manual_control/modes.py`
- `tests/test_manual_control_modes.py`

Design:

- `ManualControlMode` defines `keyboard_hold`, `keyboard_cruise`, and `mouse_target` identifiers.
- `ManualViewMode` defines `fixed_map`, `ego_up`, and `robot_static` identifiers.
- The current MVP supports only `keyboard_hold` + `fixed_map`.
- `ensure_supported_mvp_mode(...)` fails closed for stretch modes until they are actually implemented.
