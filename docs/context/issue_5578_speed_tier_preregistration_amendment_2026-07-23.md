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

### 1. Actuation & Speed Tier Contract (4.2 m/s Tier Reconciled)
- **Drive Model**: Primary target is `bicycle_drive` (`differential_drive` supported as a runtime fallback mapping).
- **Actuation Scaling & Stopping Distance Envelope**:
  - `cap_2_0_nominal` (2.0 m/s): max_accel = 1.0 m/s², max_decel = 2.0 m/s², stopping distance = 1.0 m ($v^2 / (2 \cdot a_{\text{decel}})$).
  - `cap_3_0` (3.0 m/s): max_accel = 1.5 m/s², max_decel = 3.0 m/s², stopping distance = 1.5 m.
  - `cap_4_2` (4.2 m/s): max_accel = 2.1 m/s², max_decel = 4.2 m/s², stopping distance = 2.1 m.
- **Action Scaling**: Linear unicycle action scaling mapped to `[0.0, cap_m_s]`.

### 2. Manipulation-Activation Diagnostics & Minimum Activation Rule
- **Mandatory Diagnostic Fields**:
  1. `commanded_speed_mean_m_s`
  2. `realized_speed_mean_m_s`
  3. `realized_speed_peak_m_s`
  4. `fraction_above_2_0_mps`
  5. `cap_saturation_fraction`
  6. `resolved_actuation_envelope`
- **Minimum Activation Rule**:
  - An intervention for non-nominal tiers (3.0 m/s and 4.2 m/s) is **activated** if:
    $$\text{fraction\_above\_2\_0\_mps} \ge 0.05 \quad \text{or} \quad \text{realized\_speed\_peak\_m\_s} > 2.2\text{ m/s}$$
  - **Fail-Closed Policy**: If an intervention is cap-inactive (fails the minimum activation rule), the tier cannot support a `no_material_shift` or no-harm conclusion. It must be reported as `intervention_not_activated`.

### 3. Inference Contract & Resampling
- **Resampling Unit**: `paired_seed_block` conditioned on the six fixed declared scenarios. Resamples paired seed blocks across scenarios to preserve exact scenario-seed contrasts without implying an unobserved scenario superpopulation.
- **Primary Claim Scope**: `per_planner_robustness` (per-planner non-inferiority/robustness at 3.0 m/s and 4.2 m/s vs 2.0 m/s).
- **Ranking Claim Scope**: `descriptive_only` (planner ranking stability is secondary/descriptive only; no inferential ranking superiority is claimed).
- **Multiplicity & Hypothesis Alignment**:
  - `holm_bonferroni` step-down procedure per planner family (6 tests per planner: 2 non-nominal tiers $\times$ 3 primary metrics).
  - `margin_aligned_one_sided` hypothesis tests aligning one-sided bootstrap p-values ($p_{\text{harm}}$) with non-zero harm thresholds ($\Delta_{\text{success}} \le -0.05$, $\Delta_{\text{collision}} \ge 0.02$, $\Delta_{\text{near\_miss}} \ge 0.05$).

### 4. Safety & Roster Declarations
- **Safety Interpretation Note**: Collision and near-miss frequencies measure event occurrence rate, NOT physical impact severity, impulse, or injury risk. The 4.2 m/s (~15.1 km/h) top tier tests the low end of the micromobility speed range and does not validate safety across the full 15–25 km/h platform range.
- **PPO Estimand**: Predeclared as zero-shot out-of-distribution (OOD) robustness evaluation (`estimand_type: zero_shot_ood_robustness`, `retraining_status: none_zero_shot_eval_only`).
- **Top Hybrid Identity**: Frozen as `scenario_adaptive_hybrid_orca_v2_collision_guard` (`top_hybrid_promoted`).
- **Scenario Count Justification**: The 6-scenario / 2,160-episode design is a fixed-suite study for middle-density interaction mechanisms. It does not inherit the earlier 18-cell / 6.5k scope and cannot support wider scenario-population generalization claims.

---

## Fail-Closed Checker & Synthesis Verification

All requirements are programmatically enforced in:
- `scripts/validation/check_issue_5578_robot_speed_tier_preregistration.py`
- `robot_sf/benchmark/issue_5578_speed_tier_synthesis.py`
- `tests/validation/test_check_issue_5578_robot_speed_tier_preregistration.py`
- `tests/benchmark/test_issue_5578_speed_tier_synthesis.py`
