<!-- AI-GENERATED (#8045) - NEEDS-REVIEW -->
# Planner Development Funnel and Selection Trace

<!-- schema: planner_development_summary.v1 -->

## 1. Dissertation-Facing Compact Funnel View

This view defines the standard evidence tier transitions without conflating exploratory search with benchmark claims.

| Stage | Purpose | Typical Proof | Admissible Conclusion | Separation from Final Campaign |
| --- | --- | --- | --- | --- |
| **1. Candidate Generation** | Explore navigation mechanisms and prototypes | Config / method card | Idea or implementation exists | Not evidence |
| **2. Smoke & Nominal Sanity** | Reject broken or unviable candidates | Deterministic smoke test | Executable under bounded fixture | Not ranking evidence |
| **3. Diagnostic & Stress Studies** | Identify mechanism failure modes | Diagnostic artifact / run logs | Mechanism-specific observation | Not pooled with release |
| **4. Roster Freeze** | Fix exact planner configuration identities | Hash-pinned release manifest | Experiment definitions locked | Precedes release execution |
| **5. Release Campaign** | Evaluate frozen roster on benchmark splits | 20,160-row release bundle | Authoritative dissertation results | Published result surface |

## 2. Frozen 14-Arm Release Roster Trace

Verified against release manifest `paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml` (total 14 arms).

| Key | Display Name | Family | Selection Trace | Selection Bias / Separation Accounting |
| --- | --- | --- | --- | --- |
| `prediction_planner` | **Prediction MPC Planner** | `predictive_mpc` | Primary predictive control candidate with dynamic obstacle forecasting. | `same_family_but_distinct_seeds_proven` |
| `goal` | **Direct Goal Following** | `baseline_kinematic` | Null-avoidance lower baseline. | `held_out_release_surface_proven` |
| `social_force` | **Helbing-Molnar Social Force Model** | `force_field` | Canonical classical microscopic force baseline. | `held_out_release_surface_proven` |
| `orca` | **Optimal Reciprocal Collision Avoidance** | `velocity_obstacle` | Standard multi-agent reciprocal collision avoidance baseline. | `held_out_release_surface_proven` |
| `ppo` | **Feed-Forward PPO Policy** | `reinforcement_learning` | Standard feed-forward model-free RL baseline. | `same_family_but_distinct_seeds_proven` |
| `socnav_sampling` | **Social Navigation Sampling** | `sampling_based` | Trajectory rollout sampling baseline. | `held_out_release_surface_proven` |
| `sacadrl` | **SA-CADRL Collision Avoidance** | `reinforcement_learning` | Socially-aware value network baseline. | `held_out_release_surface_proven` |
| `scenario_adaptive_hybrid_orca_v1` | **Scenario-Adaptive Hybrid ORCA v1** | `hybrid_rule` | Adaptive switching between rule-based modes and reciprocal avoidance. | `partially_overlapping_surface_disclosed` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | **Scenario-Adaptive Hybrid ORCA v2 (Collision Guard)** | `hybrid_rule` | Enhanced safety filter and emergency brake arbitration over v1. | `partially_overlapping_surface_disclosed` |
| `hybrid_rule_v3_fast_progress_static_escape` | **Hybrid Rule v3 Fast Progress (Discrete)** | `hybrid_rule` | High-efficiency rule arbitration with static obstacle escape maneuvers. | `partially_overlapping_surface_disclosed` |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | **Hybrid Rule v3 Fast Progress (Continuous)** | `hybrid_rule` | Continuous-action formulation of static escape and fast progress rules. | `partially_overlapping_surface_disclosed` |
| `guarded_ppo` | **Guarded PPO Policy** | `hybrid_learning` | Learned policy shielded by deterministic kinematic safety filter. | `partially_overlapping_surface_disclosed` |
| `predictive_mppi` | **Predictive MPPI Controller** | `sampling_based` | Model Predictive Path Integral control using sampled trajectory perturbations. | `same_family_but_distinct_seeds_proven` |
| `risk_dwa` | **Risk-Aware Dynamic Window Approach** | `dynamic_window` | Risk-weighted dynamic window velocity evaluation successor. | `partially_overlapping_surface_disclosed` |

## 3. Exploratory and Diagnostic Candidates (Predecessors / Exclusions)

Candidates developed or evaluated during exploratory phases that did not enter the frozen release roster.

| Candidate | Family | Highest Stage | Relationship | Disposition Reason |
| --- | --- | --- | --- | --- |
| `dwa_classic` | `dynamic_window` | `diagnostic_stress` | `family_represented_by_successor` | Superseded by risk_dwa with dynamic obstacle risk scoring. |
| `chance_constrained_mpc_gmm` | `predictive_mpc` | `diagnostic_stress` | `diagnostic_only` | High computational latency during dense crowd stress tests. |
| `diffusion_policy_smoke` | `generative_policy` | `smoke_nominal` | `blocked_or_unavailable` | Real-time execution latency exceeded benchmark timeout budget. |

## 4. Post-Anchor Candidate Demarcation

Planners introduced after the frozen `0.0.3.post1` dissertation evidence anchor. These candidates are strictly segregated from the historical benchmark roster.

| Post-Anchor Candidate | Family | Evidence Tier | Status |
| --- | --- | --- | --- |
| `anisotropic_gaussian_human_cost_planner` | `predictive_human_cost` | `diagnostic_only` | `post_anchor` (Strictly excluded from release roster) |
| `force_coupled_potential_field` | `potential_field` | `diagnostic_only` | `post_anchor` (Strictly excluded from release roster) |
| `recurrent_ppo_stateful_adapter` | `reinforcement_learning` | `diagnostic_only` | `post_anchor` (Strictly excluded from release roster) |

## 5. Methodological Separation Summary

- **Prospective Freeze**: The 14 release planners were frozen prior to running the 20,160-episode release matrix.
- **No Post-Hoc Roster Alteration**: Exploratory candidates (e.g. GMM chance-constrained MPC) and post-anchor planners (e.g. RecurrentPPO, human-cost Gaussian) are never backported into the dissertation release bundle.
- **Honest Bias Accounting**: Scenarios shared between diagnostic tuning and evaluation are disclosed as `partially_overlapping_surface_disclosed` rather than claiming artificial prospective isolation.
