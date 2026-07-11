# Issue #4973 — Optional Zanlungo collision-prediction pedestrian force

This opt-in simulator model lets pedestrians react to predicted closest approaches while leaving
the default pedestrian dynamics unchanged. It is an implementation prototype, not evidence that
the model is calibrated for Robot SF or that it eliminates freezing across benchmark scenarios.

## Runtime contract

Set `simulation_config.pedestrian_model` to
`hsfm_zanlungo_collision_prediction_v1`. The selector replaces only the existing
pedestrian-to-pedestrian `SocialForce` contribution; goal, obstacle, group, and robot forces remain
in the total. If the baseline social component is unavailable, the opt-in path fails closed instead
of silently changing the force composition.

The pure helper in `robot_sf/sim/pedestrian_model_variants.py` follows Zanlungo, Ikeda, and Kanda
(2011), equations 10–11:

1. For each pedestrian, find the earliest eligible straight-line closest-approach time among
   neighbors within the paper's `pi / 4` approach cone.
2. Project every neighbor to that one common time. This makes the interaction non-additive: a nearer
   conflict changes how the same pedestrian responds to all other neighbors.
3. Scale projected-distance repulsion by the pedestrian's speed divided by that time and by the
   paper's anisotropy weight.

The canonical opt-in packet is
`configs/research/zanlungo_collision_prediction_issue_4973.yaml`. Its interaction strength (`1.13`),
range (`0.71 m`), and anisotropy (`0.29`) reproduce the paper's trajectory-calibrated values. They
are not calibrated to Robot SF scenarios. A deterministic current-separation fallback resolves the
equation's exactly centered zero-distance direction, and `max_force` bounds numerical acceleration.

## Compatibility and evidence boundary

- `social_force_default` remains the default and does not read the new config.
- The existing `hsfm_ttc_predictive_v1` Karamouzas-style pairwise force is separate; it does not use
  the Zanlungo common-time, non-additive projection.
- Focused tests prove pure-force geometry, common-time coupling, deterministic degeneracy handling,
  scenario-config loading, default compatibility, runtime composition, and fail-closed behavior.
- Corridor-level freeze/deadlock acceptance remains a separate predeclared CPU comparison with
  parameter sensitivity. No benchmark campaign, Slurm/GPU job, or paper/dissertation claim is part
  of this implementation slice.

Reference: F. Zanlungo, T. Ikeda, and T. Kanda, “Social force model with explicit collision
prediction,” *EPL* 93 (2011) 68005, <https://doi.org/10.1209/0295-5075/93/68005>.
