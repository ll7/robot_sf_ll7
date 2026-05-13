# Issue #1152 Manual-Control Mode Experiments

## Goal

Issue #1152 adds the first post-MVP manual-control experiment bundle on top of the #1151
foundation: versioned steering modes, view-mode selection metadata, deterministic mapper behavior,
and artifact fields that keep manual-control recordings filterable by mode.

## Implementation boundary

The implemented steering modes are pure differential-drive mappers:

- `keyboard_cruise_diff_drive_v1`: runners keep a persistent target velocity and the mapper applies
  deterministic key increments plus explicit brake/reset.
- `mouse_target_diff_drive_v1`: runners pass a local mouse/click target and the mapper converts it
  into bounded linear/angular velocity deltas.

The repository now exposes a compact runtime config surface for `--control-mode`, `--view-mode`,
and `--robot-action-space` through `scripts/manual_control/validate_modes.py`. This gives future
interactive runners the same validated selector path without making the pure mapper tests depend on
Pygame.

## Ego-up blocker

`ego_up_view_v1` is registry-visible but intentionally fails closed. The stacked #1151 foundation
contains pure mode, recording, replay, and export primitives, but it does not yet expose an
interactive renderer camera-transform hook that can make a robot-centered, robot-up view executable.

The current blocker text is stored in `robot_sf/manual_control/modes.py` and is surfaced through
`ManualControlRuntimeConfig.from_strings(view_mode="ego_up")`. A later runner/rendering issue should
replace the fail-closed state with an implemented `ManualViewModeSpec` only after the camera
transform is exercised by a real or headless rendering smoke.

## Validation path

Targeted proof for this issue should cover:

- unit tests for hold/cruise/mouse mapper behavior,
- mode-registry and unsupported action-space validation,
- session manifest and JSONL recording mode metadata,
- CLI mode-selection smoke via `scripts/manual_control/validate_modes.py`.

Before PR handoff, run the targeted manual-control tests and then the repository PR readiness check
against the stacked #1151 base.
