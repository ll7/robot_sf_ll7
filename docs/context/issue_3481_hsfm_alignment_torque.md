# Issue #3481: HSFM Body-Orientation Alignment Torque

This note records the opt-in body-orientation **alignment-torque** slice for issue
[#3481](https://github.com/ll7/robot_sf_ll7/issues/3481). It is the maintainer-named remaining
code prerequisite ("HSFM body-orientation/alignment-torque prerequisite", 2026-07-05 gate comment)
for the Definition-of-Done bullet:

> HSFM heading state + alignment-torque term added, decoupling `phi_i` from instantaneous `v_i`.

Evidence tier: **diagnostic / prototype**. No calibrated-realism, benchmark-strength, planner-ranking,
paper-facing, or dissertation claim is made or promoted here. Sibling slice:
[`issue_3481_hsfm_ttc_predictive_forces.md`](issue_3481_hsfm_ttc_predictive_forces.md).

## Motivation: what "decoupled" means

The Phase 1 `hsfm_total_force_v1` path (and its TTC/FoV descendants) orients each pedestrian by
**snapping** the heading to the instantaneous total-force direction every step
(`arctan2(force_y, force_x)` in `step_hsfm_total_force`). That is still fully *coupled* to the
instantaneous force/velocity: the body can flip orientation arbitrarily fast.

In the Headed Social Force Model (Farina et al., 2017) the body orientation `phi_i` is a genuine
state variable driven toward the desired direction by a damped rotational torque with bounded turn
rate. This removes the instantaneous heading snap — the orientation *lags* the desired direction —
which is the behavior the issue wants so a planner cannot exploit an instant heading flip.

## Selector key

- `hsfm_alignment_torque_v1`: reuses the `hsfm_total_force_v1` position/velocity stepping, but
  replaces the instant heading snap with a damped second-order alignment torque. The total-force
  direction becomes the *desired* orientation; the actual body orientation relaxes toward it.

The default pedestrian model and all prior selectors are unchanged; a regression test pins that
`hsfm_total_force_v1` still snaps its heading.

## Torque contract

Pure, simulator-independent helpers in `robot_sf/sim/pedestrian_model_variants.py`:

- `wrap_to_pi(angle)` — wrap to `(-pi, pi]` (the `-pi` boundary folds to `+pi`).
- `step_alignment_torque_heading(headings, angular_velocities, target_headings, *, dt, k_theta,
  k_omega, max_angular_speed)` — one semi-implicit Euler step of the per-pedestrian dynamics:

```text
e      = wrap_to_pi(target - phi)                         # shortest signed angular error
omega' = clip(omega + dt * (k_theta*e - k_omega*omega),   # damped restoring torque
              -max_angular_speed, +max_angular_speed)      # bounded turn rate
phi'   = wrap_to_pi(phi + dt * omega')
```

Returns updated `(headings, angular_velocities)`. Critical damping is `k_omega = 2*sqrt(k_theta)`.

## Parameters

`SimulationSettings.alignment_torque` (`AlignmentTorqueConfig`) stores the opt-in surface, validated
fail-closed in `__post_init__`:

- `enabled` (auto-set `True` when `pedestrian_model == hsfm_alignment_torque_v1`)
- `k_theta > 0` (default `4.0`)
- `k_omega >= 0` (default `4.0`, i.e. critical damping for the default stiffness)
- `max_angular_speed_rad_s > 0` (default `pi`)

Scenario `simulation_config` can select the model and override params (see
`_set_simulation_override_attr` in `robot_sf/training/scenario_loader.py`, now table-driven over the
opt-in force-config registry).

## Simulator seam

`Simulator._step_pedestrians` tracks a per-pedestrian `ped_angular_velocities` state (zeroed at init
and on episode reset). For `hsfm_alignment_torque_v1` it calls `step_hsfm_total_force` for the
kinematics + desired heading, then integrates the actual heading with
`step_alignment_torque_heading`. Deterministic for fixed initial state and config.

## Validation

```bash
uv run pytest tests/sim/test_hsfm_alignment_torque_model.py -q
uv run pytest tests/sim/ tests/test_scenario_loader_overrides.py \
  tests/training/test_scenario_loader.py -q
uv run ruff check robot_sf/sim robot_sf/training/scenario_loader.py \
  tests/sim/test_hsfm_alignment_torque_model.py
```

`tests/sim/test_hsfm_alignment_torque_model.py` covers: `wrap_to_pi`; hold-when-aligned;
no-snap-in-one-step (decoupling); a per-element oracle match; convergence under critical damping;
angular-speed cap; shortest-turn wrapping; fail-closed parameter/shape/finiteness validation; config
validation and selector-enable provenance; scenario selection; and a deterministic simulator smoke
contrasting the lagged heading against the still-snapping `hsfm_total_force_v1`.

## Remaining for closure (unchanged by this slice)

- Seed-controlled campaign / benchmark-strength synthesis for narrow-passage lateral-sliding and
  bottleneck freeze/deadlock (campaign RUN — out of scope for CPU-only slices).
- Calibrated-realism support (needs external-data calibration).
- Any evidence-tier upgrade above diagnostic/prototype.
