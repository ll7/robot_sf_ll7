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
  --mode run \
  --label issue697_holonomic_sf_diagnosis
```

For the upstream socialforce wrapper, the `socialforce==0.2.3` package must be installed:
```bash
uv run --with socialforce==0.2.3 python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/holonomic_social_force_diagnosis.yaml \
  --mode run \
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

## Execution Results (Issue #811)

### Campaign Commands

```bash
# Local planners (upstream fails-fast without socialforce package):
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/holonomic_social_force_diagnosis.yaml \
  --mode run \
  --label issue697_holonomic_sf_diagnosis

# Upstream wrapper (with socialforce==0.2.3):
uv run --with socialforce==0.2.3 python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/holonomic_social_force_diagnosis.yaml \
  --mode run \
  --label issue697_holonomic_sf_diagnosis_upstream
```

Campaign IDs (2026-04-13):
- Local: `holonomic_social_force_diagnosis_issue697_holonomic_sf_diagnosis_20260413_134959`
- Upstream: `holonomic_social_force_diagnosis_issue697_holonomic_sf_diagnosis_upstream_20260413_135143`

Scenario: `planner_sanity_matrix_v1` → `planner_sanity_simple` (open map, no pedestrians, 3 seeds × horizon 150).

### Metric Table

| Planner                          | success | collisions | jerk   | path_efficiency | notes                        |
| -------------------------------- | ------- | ---------- | ------ | --------------- | ---------------------------- |
| `social_force` (tau=0.5, rep=0.8) | 0.0000 | 0.0000     | 2.2015 | 0.6924          | all 3 seeds timeout          |
| `social_force_tau_high` (tau=2.0) | 0.0000 | 0.0000     | 1.6796 | 0.9920          | reduced jerk, near-optimal path |
| `social_force_tau_low` (tau=0.2)  | 0.0000 | 0.0000     | 0.7649 | 1.0000          | significantly reduced jerk, optimal path |
| `social_force_repulsion_low` (rep=0.3) | 0.0000 | 0.0000 | 0.7263 | 1.0000          | lowest jerk, optimal path    |
| `social_navigation_pyenvs_socialforce` | n/a | n/a    | n/a    | n/a             | partial-failure (see below)  |

### Key Findings

**Finding 1 — All variants fail even on an empty open-map sanity scenario.**
The `planner_sanity_simple` map has no pedestrians and no obstacles.  All four local
social-force variants hit the 150-step horizon without reaching the goal on all 3 seeds.
Zero collisions confirms the robot is moving but not reaching the goal — consistent with
the pre-execution failure-mode classification (force integration produces insufficient
net forward motion in holonomic mode).

**Finding 2 — Parameter tuning reduces jerk and improves path efficiency but does not fix
goal-reaching.**
- `repulsion_low` (weight 0.8→0.3): jerk 2.2015→0.7263 (−67%), path efficiency 0.6924→1.0
- `tau_low` (tau 0.5→0.2): jerk 2.2015→0.7649 (−65%), path efficiency 0.6924→1.0
- `tau_high` (tau 0.5→2.0): jerk 2.2015→1.6796 (−24%), path efficiency 0.6924→0.992

The repulsion_low and tau_low variants move along the optimal path (`path_efficiency=1.0`)
but still time out.  This means the stall is partially a **speed/throughput issue** on
top of the oscillation hypothesis: even moving along the correct path, the effective
forward speed is insufficient to reach the goal within 150 steps.

**Finding 3 — Upstream wrapper cannot run in holonomic passthrough mode.**
With `socialforce==0.2.3` installed, the upstream `social_navigation_pyenvs_socialforce`
planner passes preflight but produces `partial-failure` at runtime.  The campaign report
shows it runs as `execution_mode=adapter`, `planner_cmd=unicycle_vw`,
`projection_policy=heading_safe_velocity_to_unicycle_vw` rather than the intended
`world_velocity_passthrough`.  This is not a valid holonomic comparison — the upstream
wrapper falls back to unicycle projection.  A separate issue is needed to diagnose and
fix the holonomic projection path for this wrapper before an upstream comparison can be
drawn.

### Video Artifact

```bash
uv run python scripts/tools/render_scenario_videos.py \
  --scenario configs/scenarios/planner_sanity_matrix_v1.yaml \
  --policy socnav_social_force \
  --all --seed 101 --max-steps 150 \
  --output output/videos/issue697_diagnosis
```

Rendered: `output/videos/issue697_diagnosis/planner_sanity_simple_seed101_socnav_social_force.mp4`

Visual interpretation: the robot oscillates in place near the start position under the
default social-force parameters.  The repulsion_low and tau_low sweep variants cannot
be rendered separately with the current `render_scenario_videos.py` tool because it
does not expose custom algo-config injection for the SF adapter.  A follow-up is needed
to render sweep-variant trajectories.

### Outcome Decision

All four local social-force variants fail on the simplest sanity scenario.  Parameter
tuning helps jerk and path efficiency but does not recover goal-reaching.  The
upstream wrapper comparison is inconclusive (unicycle fallback, not holonomic).

**Decision: document `social_force` as inherently limited in holonomic mode.**

The holonomic benchmark profile should be updated to reflect that:
- `social_force` (local) is expected to have zero success in holonomic `vx_vy` mode
- The repulsion_low variant is the best-performing sweep but still does not succeed
- The upstream wrapper holonomic projection path is broken and requires a separate fix
- No parameter in the current `SocNavPlannerConfig` range produces non-zero success

Follow-up speed-cap sweep on 2026-04-13 reinforced the conclusion: raising
`max_linear_speed` and `social_force_desired_speed` to `1.5` improved
`path_efficiency` only slightly on `planner_sanity_simple` (`0.9926` vs `1.0000` for
`repulsion_low`) and still yielded `success=0.0000` for all 3 seeds.

## Validation / Follow-up

- [x] Run `holonomic_social_force_diagnosis.yaml` and record metrics per planner key.
- [x] Run upstream wrapper with `--with socialforce==0.2.3` for the direct comparison.
- [x] Render at least one short diagnostic video via
  `scripts/tools/render_scenario_videos.py` to visualize the stall/oscillation pattern.
- [x] Update this note with actual metric rows and video interpretation.
- [x] Run a follow-up speed-cap sweep that raised `max_linear_speed` and
  `social_force_desired_speed` to `1.5` and verify it still did not recover success.
- [ ] If a parameter sweep shows clear improvement, open a follow-up issue to promote the
  tuned config and update the holonomic benchmark entry.
- [x] If all variants fail similarly, update the holonomic benchmark profile to document
  `social_force` as inherently weak in holonomic mode and note the upstream wrapper result.

### Deferred Follow-up Items

- **Upstream holonomic projection fix**: the `social_navigation_pyenvs_socialforce` wrapper
  does not route through world-velocity passthrough in holonomic mode — it falls back to
  `heading_safe_velocity_to_unicycle_vw`.  A separate issue is needed to diagnose whether
  the algo config `projection_policy` field is not wired to the execution path for this
  wrapper.
- **Sweep-variant video rendering**: `render_scenario_videos.py` does not support injecting
  a custom `SocNavPlannerConfig` for SF parameter variants.  A follow-up enhancement would
  add `--socnav-tau` and `--socnav-repulsion-weight` flags to allow direct visual comparison.
