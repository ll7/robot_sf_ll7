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

## Ego-up renderer hook

Issue #1604 removes the `ego_up_view_v1` fail-closed blocker by adding a renderer-facing camera
hook to `SimulationView` and replay support through `LazySimulationView`. Manual-control sessions
now configure the environment renderer with `set_manual_view_mode("ego_up")`, which centers the
camera on the robot and rotates world coordinates so the robot heading points toward screen up.

The fixed-map path remains backward-compatible: renderers without a manual camera hook are allowed
for `fixed_map`, and `SimulationView` falls back to the original scale-plus-offset transform when no
ego-up camera center/rotation is active. `robot_static` remains intentionally fail-closed until a
separate renderer mode is implemented.

## Validation path

Targeted proof for this issue should cover:

- unit tests for hold/cruise/mouse mapper behavior,
- mode-registry and unsupported action-space validation,
- session manifest and JSONL recording mode metadata,
- CLI mode-selection smoke via `scripts/manual_control/validate_modes.py`.
- for issue #1604 specifically: ego-up mode-registry support, unsupported-renderer fail-closed
  behavior, `LazySimulationView` replay/forwarding, manual runner renderer wiring, and the
  `SimulationView` transform that maps robot heading to screen up.

Before PR handoff, run the targeted manual-control tests and then the repository PR readiness check
against the current `origin/main` base.

Issue #1604 targeted validation on 2026-05-30:

- `uv run pytest -q tests/test_manual_control_modes.py tests/test_manual_control_pygame_runner.py tests/test_lazy_pygame_init.py tests/visuals/test_sim_view_coverage_paths.py::test_sim_view_ego_up_camera_transform_centers_robot_and_rotates_heading tests/visuals/test_sim_view_coverage_paths.py::test_sim_view_default_camera_transform_keeps_existing_affine_scaling`
  passed with `34 passed`.
- `uv run ruff check robot_sf/manual_control robot_sf/render tests/test_manual_control_modes.py tests/test_manual_control_pygame_runner.py tests/test_lazy_pygame_init.py tests/visuals/test_sim_view_coverage_paths.py`
  passed.
