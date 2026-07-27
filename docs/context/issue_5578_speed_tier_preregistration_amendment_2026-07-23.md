# Issue #5578 Robot Speed-Tier Preregistration Amendment (Issue #6100)

Date: 2026-07-23
Governance Parent: #5557
Preregistration Parent: #5578
Amendment Decision Issue: #6100

---

## Executive Summary

This amendment freezes the final run-ready protocol contract for the issue #5578 robot speed-tier benchmark sweep prior to any registered campaign execution. Per maintainer decisions in #6100, this document and the updated canonical preregistration (`configs/benchmarks/issue_5578_robot_speed_tier_preregistration.yaml`) resolve all open intervention, inference, and safety-interpretation questions.

---

## Key Decisions & Frozen Contracts

### 1. Actuation & Speed Tier Contract (Supported 4.0 m/s Top Tier)

- **Drive Model**: Every registered tier uses the exact `bicycle_drive` runtime variants in `configs/research/robot_speed_band_v1.yaml`.
- **Top-tier resolution**: The runtime reference has no exact 4.2 m/s variant. Following #6100's fail-closed stop rule, this amendment uses the already supported `bicycle_4_0_mps_micromobility` variant rather than inventing or claiming a 4.2 m/s binding.
- **Actuation Scaling & Stopping Distance Envelope**:
  - `cap_2_0_nominal` (2.0 m/s): max_accel = 1.0 m/s², max_decel = 2.0 m/s², stopping distance = 1.0 m ($v^2 / (2 \cdot a_{\text{decel}})$).
  - `cap_3_0` (3.0 m/s): max_accel = 1.5 m/s², max_decel = 3.0 m/s², stopping distance = 1.5 m.
  - `cap_4_0` (4.0 m/s): max_accel = 2.0 m/s², max_decel = 4.0 m/s², stopping distance = 2.0 m.
- **Two-stage production control path**:
  1. Planners emit physical `unicycle_vw` commands, not normalized actions.
     Linear velocity is bounded to `[0.0, cap_m_s]`. Angular velocity is
     bounded to
     `[-cap_m_s * tan(0.78) / 1.0, +cap_m_s * tan(0.78) / 1.0]`, yielding
     approximately ±1.9785, ±2.9678, and ±3.9570 rad/s for the three tiers.
  2. The campaign runner's production `_env_action` converts target speed and
     yaw rate into the bicycle environment's native
     `[acceleration, steering_angle]` action. It clips target speed to the tier,
     computes acceleration as
     `(target_speed - current_speed) / max(dt, 1e-6)`, computes steering as zero
     below the target-speed epsilon or
     `atan(omega * wheelbase / max(abs(target_speed), 1e-6)) * sign(target_speed)`,
     and finally clips to the environment action space.
- **Frozen native parameters**: `wheelbase=1.0 m`, `max_steer=0.78 rad`,
  `dt=0.1 s`; native acceleration bounds are `[-max_decel, +max_accel]`, and
  steering bounds are `[-0.78, +0.78]`.
- **PPO adapter boundary**: the zero-shot PPO baseline is separately bound to
  `ppo_action_to_unicycle`. Its checked-in configuration emits physical
  `unicycle_vw` commands bounded to `[0.0, 2.0] m/s` and `[-1.0, 1.0] rad/s`;
  there is no `[0,1]` normalization and no silent tier rescaling. Consequently,
  the activation gate may correctly classify PPO's higher tiers as
  `intervention_not_activated`.

### 2. Manipulation-Activation Diagnostics & Minimum Activation Rule
- **Mandatory Diagnostic Fields**:
  1. `commanded_speed_mean_m_s`
  2. `realized_speed_mean_m_s`
  3. `realized_speed_peak_m_s`
  4. `fraction_above_2_0_mps`
  5. `cap_saturation_fraction`
  6. `resolved_actuation_envelope`
- **Minimum Activation Rule**:
  - An intervention for non-nominal tiers (3.0 m/s and 4.0 m/s) is **activated** if:
    $$\text{fraction\_above\_2\_0\_mps} \ge 0.05 \quad \text{or} \quad \text{realized\_speed\_peak\_m\_s} > 2.2\text{ m/s}$$
  - **Fail-Closed Policy**: If an intervention is cap-inactive (fails the minimum activation rule), the tier cannot support a `no_material_shift` or no-harm conclusion. It must be reported as `intervention_not_activated`.

### 3. Inference Contract & Resampling
- **Resampling Unit**: `paired_seed_block` conditioned on the six fixed declared scenarios. Resamples paired seed blocks across scenarios to preserve exact scenario-seed contrasts without implying an unobserved scenario superpopulation.
- **Primary Claim Scope**: `per_planner_robustness` (per-planner non-inferiority/robustness at 3.0 m/s and 4.0 m/s vs 2.0 m/s).
- **Ranking Claim Scope**: `descriptive_only` (planner ranking stability is secondary/descriptive only; no inferential ranking superiority is claimed).
- **Multiplicity & Hypothesis Alignment**:
  - Six estimands per planner (2 non-nominal tiers $\times$ 3 primary metrics).
  - Two predeclared directional Holm families per planner: material harm and non-inferiority. The familywise alpha of 0.05 is split equally (`0.025` per directional family), so making either directional claim retains a combined 0.05 error budget.
  - Each comparison uses a margin-aligned one-sided paired-seed-block percentile-bootstrap bound and a plus-one empirical tail probability at the non-zero harm threshold ($\Delta_{\text{success}} \le -0.05$, $\Delta_{\text{collision}} \ge 0.02$, $\Delta_{\text{near\_miss}} \ge 0.05$). No two-sided zero-effect p-value orders these decisions.

### 4. Safety & Roster Declarations
- **Safety Interpretation Note**: Collision and near-miss frequencies measure event occurrence rate, NOT physical impact severity, impulse, or injury risk. The supported 4.0 m/s (~14.4 km/h) top tier is a micromobility-direction stress test below 15 km/h and does not validate safety across the 15–25 km/h platform range.
- **PPO Estimand**: Predeclared as zero-shot out-of-distribution (OOD) robustness evaluation (`estimand_type: zero_shot_ood_robustness`, `retraining_status: none_zero_shot_eval_only`).
- **Top Hybrid Identity**: Frozen as `scenario_adaptive_hybrid_orca_v2_collision_guard` (`top_hybrid_promoted`).
- **Scenario Count Justification**: The 6-scenario / 2,160-episode design is a fixed-suite study for middle-density interaction mechanisms. It does not inherit the earlier 18-cell / 6.5k scope and cannot support wider scenario-population generalization claims.

---

## Fail-Closed Checker & Synthesis Verification

The checked-in protocol, row parser, and focused tests fail closed on missing
actuation/exposure diagnostics, runtime-tier drift, planner-command or native
action-bound drift, conversion-formula drift, PPO adapter drift, numeric
threshold drift, incomplete grids, and claim-boundary drift. The checker reads
the production runtime variant configuration and live bicycle/simulation
defaults; real-path tests instantiate the environment and execute the
`unicycle_vw` → `_env_action` conversion at all three tiers. They do not
constitute campaign evidence. The enforcement surfaces are:
- `scripts/validation/check_issue_5578_robot_speed_tier_preregistration.py`
- `robot_sf/benchmark/issue_5578_speed_tier_synthesis.py`
- `tests/validation/test_check_issue_5578_robot_speed_tier_preregistration.py`
- `tests/benchmark/test_issue_5578_speed_tier_synthesis.py`
