<!-- AI-GENERATED NEEDS-REVIEW -->
# Dissertation Coverage Aggregate

This is a repository-side provenance map for downstream review. It preserves the named source packages and their status wording; it is not benchmark evidence and does not edit or read a dissertation repository.

## Frozen consumer profile

- Profile: `configs/publication/dissertation_coverage_v1.yaml`
- Consumer: `ll7-diss-submission-2026`
- Repository: `ll7/robot_sf_ll7`
- Anchor source commit: `b1d5ab6de708385c0828c99501a9d1c29727ec11`
- Release tag: `paper-matrix-v2-h600-s30-2026-08-cd831d7582c1`
- DOI: `10.5281/zenodo.22077448`
- Campaign identity: `paper_matrix_v2_h600_s30_2026_08_cd831d7582c1`
- Frozen matrix identity: 14 planners × 48 scenario cells × 30 seeds = 20160 expected episodes.

## Source packages

| Source | Producer | Schema | Records | SHA-256 |
| --- | --- | --- | ---: | --- |
| `docs/context/evidence/future_work_cards/amv_actuation_realism_bridge.v1.json` | issue #8048 / PR #8137 (55e6f91491f4) | `future_work_bridge_status_card.v1` | 1 | `1ae39ff43de5456a0418faeb2a897972dac37c33b53dd5390ce9b1d98b7f71b4` |
| `docs/context/evidence/future_work_cards/carla_cross_simulator_bridge.v1.json` | issue #8048 / PR #8137 (55e6f91491f4) | `future_work_bridge_status_card.v1` | 1 | `734e8382d47310d14517ce80187653a6e282014a6fe90b8df96e63fa85257b81` |
| `docs/context/evidence/future_work_cards/incident_to_scenario_provenance.v1.json` | issue #8048 / PR #8137 (55e6f91491f4) | `future_work_bridge_status_card.v1` | 1 | `2c02a7174b732fbe924a982abac11c177b750b73e9925861590dfba9263ec181` |
| `docs/context/evidence/future_work_cards/route_choice_homotopy_observability.v1.json` | issue #8048 / PR #8137 (55e6f91491f4) | `future_work_bridge_status_card.v1` | 1 | `42d377e43cd50a8b276880e38f07fab17e620e040685549a324fcd73dd30a54b` |
| `docs/context/evidence/planner_development_funnel.v1.json` | issue #8045 / PR #8142 (f5eb7af66469) | `planner_development_funnel.v1` | 20 | `c2f9c365fe1583d256f84a698f99ffc12f956df3754632c2020d3637375589ad` |
| `docs/context/evidence/post_anchor_capability_delta.v1.json` | issue #8046 / PR #8138 (31d40a3d0930) | `post_anchor_capability_delta.v1` | 12 | `5941b37252a952b0a5201828b110f3112601de141494d6f304e4b35df5ce1e5d` |

## Explicit source reconciliation

- Planner roster: `conflict_explicitly_recorded`. Source-only keys: `scenario_adaptive_hybrid_orca_v1`; current release-manifest-only keys: `scenario_adaptive_hybrid_orca_v2_bottleneck_yield`.
- Owner paths: `known_stale_paths_explicitly_recorded`. Missing paths named by the source: `docs/context/carla_replay_parity.md, scripts/carla/, robot_sf/provenance/, scripts/carla/`.
- These discrepancies are retained as source-accounting facts. They are not repaired, inferred away, or used to promote evidence.

## Coverage counts

- By anchor relation: `introduced_after_anchor`=7, `operational_only`=4, `predecessor_only`=3, `present_at_anchor`=15, `unknown`=1.
- By evidence status: `diagnostic_only`=7, `operational_only`=4, `release_evaluated`=14, `smoke_only`=1, `synthetic_fixture`=3, `unsupported_proxy`=1.
- By dissertation status: `absent`=6, `future_work_mentioned`=3, `intentionally_out_of_scope`=4, `unknown`=17.

## Capability rows

The row contract is: capability → relation to frozen dissertation anchor → implementation status → evidence status → dissertation relationship → strongest permitted wording → exact missing proof.

| Capability | Anchor relation | Implementation | Evidence | Dissertation relationship | Strongest permitted wording | Exact missing proof |
| --- | --- | --- | --- | --- | --- | --- |
| `actionlint_and_ci_workflow_ratchets`<br>Repository-Owned Actionlint and CI Workflow Ratchets | `operational_only` | `implemented` | `operational_only` | `repository_only` | Automated static validation of GitHub Action workflows and pagination bounds. | None (operational tooling). |
| `amv_actuation_realism_bridge`<br>AMV Actuation Realism Bridge | `present_at_anchor` | `prototype` | `unsupported_proxy` | `future_work_bridge` | Public longitudinal e-scooter evidence provides a bounded proxy-source basis, while platform-specific yaw, latency, dynamics, and physical calibration remain absent. | Closed-loop sim-to-real trajectory tracking validation on a physical AMV.<br>Physical hardware dynamometer and trajectory tracking measurements.<br>Physical vehicle platform system identification (measured command-to-motion latency, motor response curves).<br>Rotational dynamics, tire slip, terrain-dependent friction, and non-holonomic yaw inertia. |
| `anisotropic_gaussian_human_cost_planner`<br>Anisotropic Gaussian Human-Cost Planner | `introduced_after_anchor` | `implemented` | `diagnostic_only` | `post_anchor_candidate` | Anisotropic Gaussian human-cost planner core implemented and unit-tested; not evaluated against benchmark release suites. | Frozen model checkpoint registration and runtime profiling.<br>Not provided by this source package; consult the source before stronger wording.<br>Standardized benchmark comparison across all 4 benchmark tracks. |
| `carla_cross_simulator_bridge`<br>CARLA Cross-Simulator Bridge | `introduced_after_anchor` | `partial` | `diagnostic_only` | `future_work_bridge` | A pinned CARLA live-replay prototype exists and has demonstrated client/server connection plus bounded replay handling; matched actor-complete replay, metric parity, and cross-simulator validation remain unestablished. | Actor-complete replay parity between CARLA and Robot SF.<br>Coordinate, temporal, and action mapping formal equivalence proof.<br>Cross-simulator metric semantic parity (TTC, comfort, SocialForce force distributions).<br>Matched actor-complete cross-simulator scenario replay between Robot SF and CARLA.<br>Metric semantic equivalence proof.<br>Paired failure mode comparisons under native vs fallback execution. |
| `chance_constrained_mpc_gmm`<br>Chance-Constrained MPC (GMM) | `predecessor_only` | `unknown` | `diagnostic_only` | `not_reported_by_source` | Diagnostic research candidate; latency bounds precluded release admission. | Not provided by this source package; consult the source before stronger wording. |
| `diffusion_policy_smoke`<br>Diffusion Policy Adapter | `unknown` | `unknown` | `smoke_only` | `not_reported_by_source` | Implementation prototype; real-time inference unviable on CPU benchmark lanes. | Not provided by this source package; consult the source before stronger wording. |
| `dwa_classic`<br>Classic Dynamic Window Approach | `predecessor_only` | `unknown` | `diagnostic_only` | `not_reported_by_source` | Diagnostic exploratory baseline; not included in frozen 14-arm roster. | Not provided by this source package; consult the source before stronger wording. |
| `force_coupled_potential_field`<br>Force-Coupled Potential-Field Core and Comparator | `introduced_after_anchor` | `implemented` | `diagnostic_only` | `post_anchor_candidate` | Force-coupled potential-field planner implemented as a local navigation comparator; benchmark-grade evaluation unestablished. | Convergence and oscillation proof in dense crowds.<br>Not provided by this source package; consult the source before stronger wording.<br>Paired closed-loop evaluations on benchmark splits. |
| `function_length_and_complexity_audits`<br>Function-Length and Helper Call Attribution Audits | `operational_only` | `implemented` | `operational_only` | `repository_only` | Module-qualified identity and call attribution for static linters and config audits. | None (operational tooling). |
| `goal`<br>Direct Goal Following | `present_at_anchor` | `unknown` | `release_evaluated` | `not_reported_by_source` | Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign. | Not provided by this source package; consult the source before stronger wording. |
| `guarded_ppo`<br>Guarded PPO Policy | `present_at_anchor` | `unknown` | `release_evaluated` | `not_reported_by_source` | Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign. | Not provided by this source package; consult the source before stronger wording. |
| `hybrid_rule_v3_fast_progress_static_escape`<br>Hybrid Rule v3 Fast Progress (Discrete) | `present_at_anchor` | `unknown` | `release_evaluated` | `not_reported_by_source` | Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign. | Not provided by this source package; consult the source before stronger wording. |
| `hybrid_rule_v3_fast_progress_static_escape_continuous`<br>Hybrid Rule v3 Fast Progress (Continuous) | `present_at_anchor` | `unknown` | `release_evaluated` | `not_reported_by_source` | Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign. | Not provided by this source package; consult the source before stronger wording. |
| `incident_to_scenario_provenance`<br>Incident-to-Scenario Provenance | `introduced_after_anchor` | `implemented` | `synthetic_fixture` | `post_anchor_candidate` | A fail-closed provenance contract can distinguish source facts, extracted hypotheses, simulator assumptions, and replay identity for a synthetic incident fixture; real-report validity and representativeness remain future work. | Audited conversion accuracy from official reports to simulation maps.<br>Empirical validation that reconstructed scenarios faithfully represent real incidents.<br>Human-audited extraction accuracy and representativeness bounds.<br>Ingestion and validation of real-world public transportation or robot collision incident reports.<br>Ingestion of real public transit / robot collision records. |
| `issue_claim_and_queue_automation`<br>Agent Issue Claim and Queue Admission Tooling | `operational_only` | `implemented` | `operational_only` | `repository_only` | Autonomous issue claim lifecycle and prepublication ancestry validation active. | None (operational tooling). |
| `orca`<br>Optimal Reciprocal Collision Avoidance | `present_at_anchor` | `unknown` | `release_evaluated` | `not_reported_by_source` | Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign. | Not provided by this source package; consult the source before stronger wording. |
| `ppo`<br>Feed-Forward PPO Policy | `present_at_anchor` | `unknown` | `release_evaluated` | `not_reported_by_source` | Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign. | Not provided by this source package; consult the source before stronger wording. |
| `prediction_planner`<br>Prediction MPC Planner | `present_at_anchor` | `unknown` | `release_evaluated` | `not_reported_by_source` | Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign. | Not provided by this source package; consult the source before stronger wording. |
| `predictive_mppi`<br>Predictive MPPI Controller | `present_at_anchor` | `unknown` | `release_evaluated` | `not_reported_by_source` | Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign. | Not provided by this source package; consult the source before stronger wording. |
| `recurrent_ppo_stateful_adapter`<br>Stateful RecurrentPPO Planner Adapter | `introduced_after_anchor` | `implemented` | `diagnostic_only` | `post_anchor_candidate` | RecurrentPPO stateful observation and hidden-state handling adapter implemented; full training campaign results unverified. | Comparative performance against standard feed-forward PPO baselines.<br>Not provided by this source package; consult the source before stronger wording.<br>Trained recurrent policy checkpoint with deterministic seed verification. |
| `release_candidate_builder_and_verification`<br>Immutable Software Candidate Release Builder | `operational_only` | `implemented` | `operational_only` | `repository_only` | Hermetic candidate artifact build, extraction, and validation tooling active. | None (operational tooling). |
| `risk_dwa`<br>Risk-Aware Dynamic Window Approach | `present_at_anchor` | `unknown` | `release_evaluated` | `not_reported_by_source` | Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign. | Not provided by this source package; consult the source before stronger wording. |
| `route_choice_homotopy_observability`<br>Route Choice and Homotopy Observability | `introduced_after_anchor` | `prototype` | `synthetic_fixture` | `future_work_bridge` | The repository can deterministically classify route side and homotopy consistency on synthetic fixtures; whether those observables improve human predictability or social acceptance remains unevaluated. | Controlled user studies or real-world pedestrian interaction logs.<br>Empirical proof that visible topological features improve human trajectory prediction or perceived social comfort.<br>Human behavioral ground-truth or preference datasets validating route observability. |
| `route_side_homotopy_observability`<br>Route-Side and Homotopy Observability Diagnostics | `introduced_after_anchor` | `implemented` | `synthetic_fixture` | `post_anchor_candidate` | Deterministic route-side and topological homotopy classification verified on synthetic fixtures; human perceptual validation unevaluated. | Empirical human perceptual study data.<br>Ground-truth route preference distributions from real pedestrians. |
| `sacadrl`<br>SA-CADRL Collision Avoidance | `present_at_anchor` | `unknown` | `release_evaluated` | `not_reported_by_source` | Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign. | Not provided by this source package; consult the source before stronger wording. |
| `scenario_adaptive_hybrid_orca_v1`<br>Scenario-Adaptive Hybrid ORCA v1 | `present_at_anchor` | `unknown` | `release_evaluated` | `not_reported_by_source` | Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign. | Not provided by this source package; consult the source before stronger wording. |
| `scenario_adaptive_hybrid_orca_v2_collision_guard`<br>Scenario-Adaptive Hybrid ORCA v2 (Collision Guard) | `present_at_anchor` | `unknown` | `release_evaluated` | `not_reported_by_source` | Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign. | Not provided by this source package; consult the source before stronger wording. |
| `scenario_search_feasibility_diagnostics`<br>Feasibility-First Scenario Search Diagnostics | `predecessor_only` | `implemented` | `diagnostic_only` | `post_anchor_candidate` | Scenario search feasibility pruning diagnostics materially extended; computational efficiency improved on synthetic envelopes. | Full-scale search campaign across full scenario catalog. |
| `social_force`<br>Helbing-Molnar Social Force Model | `present_at_anchor` | `unknown` | `release_evaluated` | `not_reported_by_source` | Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign. | Not provided by this source package; consult the source before stronger wording. |
| `socnav_sampling`<br>Social Navigation Sampling | `present_at_anchor` | `unknown` | `release_evaluated` | `not_reported_by_source` | Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign. | Not provided by this source package; consult the source before stronger wording. |

## Repository-only capabilities

Operational rows remain separate from dissertation scientific findings:

| Capability | Evidence | Permitted wording |
| --- | --- | --- |
| `actionlint_and_ci_workflow_ratchets` | `operational_only` | Automated static validation of GitHub Action workflows and pagination bounds. |
| `function_length_and_complexity_audits` | `operational_only` | Module-qualified identity and call attribution for static linters and config audits. |
| `issue_claim_and_queue_automation` | `operational_only` | Autonomous issue claim lifecycle and prepublication ancestry validation active. |
| `release_candidate_builder_and_verification` | `operational_only` | Hermetic candidate artifact build, extraction, and validation tooling active. |

## Claim boundary and rebuild

This is repository-side provenance metadata only. It does not establish new benchmark performance, planner superiority, physical transfer, dissertation coverage, or a manuscript claim.

```text
uv run python scripts/analysis/build_dissertation_coverage_manifest.py --check
```
