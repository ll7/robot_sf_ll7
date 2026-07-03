# Issue #3481: Headed Social Force Model + Time-To-Collision Predictive Pedestrian Forces

This note records the Phase 2 prototype contract for issue
[#3481](https://github.com/ll7/robot_sf_ll7/issues/3481): an opt-in Headed Social Force Model
(HSFM) selector with Time-To-Collision (TTC) predictive pedestrian-pedestrian repulsion. It is
diagnostic evidence only, not a calibrated realism claim.

## Selector Keys

- `social_force_default`: existing PySocialForce stepping path.
- `hsfm_total_force_v1`: Phase 1 opt-in total-force HSFM stepping path.
- `hsfm_ttc_predictive_v1`: Phase 2 opt-in total-force path plus pedestrian-pedestrian
  time-to-collision predictive repulsion.

The default pedestrian model is unchanged.

## TTC Force Contract

Pure helpers in `robot_sf/sim/pedestrian_model_variants.py` compute pairwise time collision from
relative position, relative velocity, and summed pedestrian radii:

```text
||p_ij + v_ij t|| <= r_i + r_j
```

Pairs with no positive root inside `horizon_s` return `inf`. Active pairs add a bounded repulsion
term weight:

```text
force_scale * exp(-ttc / tau0_s)
```

The per-actor sum is capped by `max_force`.

## Parameters

`SimulationSettings.ttc_predictive_force` stores the opt-in surface:

- `tau0_s > 0`
- `horizon_s > 0`
- `force_scale >= 0`
- `max_force > 0`
- `include_ped_ped=True`
- `include_robot_proxy=False`

Robot-proxy TTC coupling intentionally fails closed when enabled because this slice does not add a
stable robot-state injection point for pedestrian stepping.

Tracked prototype metadata lives in
`configs/research/hsfm_ttc_predictive_forces_issue_3481.yaml`.

## Evidence Boundary

Focused simulator tests cover finite TTC values, inactive separating pairs, monotonic force growth
for shorter collision times, force capping, selector validation, and deterministic one-step smoke
for `hsfm_ttc_predictive_v1`.

This slice did not run a full benchmark campaign, submit Slurm/GPU work, perform external
calibration, or edit paper/dissertation claims.
