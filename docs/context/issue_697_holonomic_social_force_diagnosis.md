# Issue #697 — Holonomic Social-Force Tuning and Scenario-Difficulty Diagnosis

## Problem Context

Issue #690 introduced a direct holonomic world-velocity path for the local `social_force`
planner adapter.  The probe campaign from that issue produced the following quality shift:

| Metric      | Before (corrected holonomic) | After (direct holonomic path) |
| ----------- | ---------------------------- | ----------------------------- |
| success     | 0.0071                       | 0.0000                        |
| collisions  | 0.2411                       | 0.1135                        |
| SNQI        | −3.1917                      | −3.6294                       |
| jerk        | 0.4338                       | 1.7103                        |

Interpretation from #690: direct holonomic passthrough lowers collisions but drives success
to zero and quadruples jerk, indicating the planner is alive and avoiding pedestrians but
failing to reach its goal.

This note documents the diagnosis infrastructure added for issue #697 and records the
analysis used to classify the dominant failure mode.

## Failure Mode Classification

Based on the #690 evidence and inspection of `SocialForcePlannerAdapter` in
`robot_sf/planner/socnav.py`:

**Dominant failure mode: repulsion-dominated stall with high-frequency force oscillation.**

Supporting evidence:
- Success collapses to zero while collisions halve — the robot is deflecting around
  pedestrians but not recovering goal-directed motion.
- Jerk increases 4× — the holonomic path applies the raw social-force vector directly as
  a velocity increment each timestep.  In differential-drive mode, heading dynamics absorb
  some of this oscillation; in holonomic mode, the full force vector is applied immediately.
- The `social_force_tau = 0.5` default relaxation time creates a desired-force term
  `(desired_vel - robot_vel) / tau` that is large when the robot has been deflected and
  oscillates against the repulsion field.

This pattern is consistent with **stall/oscillation due to force integration instability** in
holonomic mode rather than a pure scenario-difficulty effect.

### Alternative hypotheses evaluated

| Hypothesis                                   | Assessment                                           |
| -------------------------------------------- | ---------------------------------------------------- |
| Scenario too hard for any planner            | Unlikely: `goal` and `orca` achieve non-zero success in the same holonomic matrix. |
| Upstream wrapper inherently better           | Unclear without comparison run; wrapper uses a different SF formulation (`socialforce==0.2.3`). |
| Repulsion weight too high                    | Plausible: default 0.8 may over-deflect in holonomic mode; the low-repulsion sweep tests this. |
| Tau too short (fast oscillation)             | Plausible: high-tau sweep (tau=2.0) should show whether smoother desired-force transients help. |
| Tau too long (slow goal recovery)            | Less likely given zero success; low-tau sweep (tau=0.2) falsifies over-damping. |

## Infrastructure Added

### Comparison Config

`configs/benchmarks/holonomic_social_force_diagnosis.yaml`

Runs the following planners side-by-side on the `planner_sanity_matrix_v1` scenario set in
holonomic `vx_vy` mode:

- `social_force` (local, default parameters)
- `social_navigation_pyenvs_socialforce` (upstream wrapper, world-velocity passthrough)
- `social_force_tau_high` (tau=2.0 sweep)
- `social_force_tau_low` (tau=0.2 sweep)
- `social_force_repulsion_low` (repulsion_weight=0.3 sweep)

Run with:
```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/holonomic_social_force_diagnosis.yaml \
  --mode preflight \
  --label issue697_holonomic_sf_diagnosis
```

For the upstream socialforce wrapper, the `socialforce==0.2.3` package must be installed:
```bash
uv run --with socialforce==0.2.3 python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/holonomic_social_force_diagnosis.yaml \
  --mode preflight \
  --label issue697_holonomic_sf_diagnosis_upstream
```

### Parameter Sweep Algo Configs

| File                                                          | Key change         | Hypothesis tested                     |
| ------------------------------------------------------------- | ------------------ | ------------------------------------- |
| `configs/algos/social_force_holonomic_tuned_tau_high.yaml`    | tau 0.5 → 2.0      | Slower desired-force transient reduces jerk |
| `configs/algos/social_force_holonomic_tuned_tau_low.yaml`     | tau 0.5 → 0.2      | Faster goal tracking recovers success |
| `configs/algos/social_force_holonomic_tuned_repulsion_low.yaml` | repulsion 0.8 → 0.3 | Weaker repulsion allows goal-seeking  |

### Upstream Wrapper Config

`configs/algos/social_navigation_pyenvs_socialforce_holonomic_probe.yaml` — algo config for
the upstream `SocialForce` policy with `projection_policy: world_velocity_passthrough`.
Added to `configs/benchmarks/holonomic_upstream_wrappers_probe.yaml` alongside the existing
orca and sfm_helbing entries.

## Upstream Wrapper Comparison Notes

The `social_navigation_pyenvs_socialforce` wrapper uses the `socialforce==0.2.3` external
library rather than `fast-pysf`.  Key differences:

- Different force formulation (Helbing-style vs. fast-pysf implementation)
- Different numerical integration (library-internal, not directly tunable through
  `SocNavPlannerConfig`)
- Same holonomic execution path: upstream `ActionXY` → world-velocity passthrough

If the upstream wrapper shows higher success than the local adapter under identical scenarios,
the failure is in the local force formulation or parameter settings rather than scenario
difficulty.  If both fail similarly, scenario difficulty is the primary explanation.

## Conservative Conclusions (Pre-Execution)

1. The dominant failure mode for local `social_force` in holonomic mode is consistent with
   **force integration instability** rather than scenario difficulty.
2. The `tau` and `repulsion_weight` parameters are the most likely tuning levers.
3. Any improvement claim requires direct execution evidence from the diagnosis config above.
4. The holonomic benchmark contract from #690 must not be weakened regardless of outcome —
   if the local social-force adapter cannot achieve non-zero success, this should be documented
   as an inherent limitation, not worked around via fallback behavior.
5. The upstream wrapper comparison is necessary to separate "tunable" from "planner mismatch"
   failure modes.

## Validation / Follow-up

- [ ] Run `holonomic_social_force_diagnosis.yaml` and record metrics per planner key.
- [ ] Run upstream wrapper with `--with socialforce==0.2.3` for the direct comparison.
- [ ] Render at least one short diagnostic video per planner variant via
  `scripts/tools/render_scenario_videos.py` to visualize the stall/oscillation pattern.
- [ ] Update this note with actual metric rows and video interpretation.
- [ ] If a parameter sweep shows clear improvement, open a follow-up issue to promote the
  tuned config and update the holonomic benchmark entry.
- [ ] If all variants fail similarly, update the holonomic benchmark profile to document
  `social_force` as inherently weak in holonomic mode and note the upstream wrapper result.
