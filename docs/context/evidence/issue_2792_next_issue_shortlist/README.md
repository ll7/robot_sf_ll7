# Next-Issue Shortlist

**Purpose**: synthesis/planning aid; not new benchmark, paper, dissertation, or safety evidence.

Generated: 2026-06-14T02:10:57.333370+00:00  | Schema: next_issue_shortlist.v1

## Data Sources

- **negative_result_register**: available [docs/context/evidence/issue_2762_negative_result_register/register.json]
- **dissertation_evidence_ledger**: available [docs/context/evidence/issue_2760_dissertation_evidence_ledger/ledger.json]
- **dissertation_gap_report**: available [docs/context/evidence/issue_2784_dissertation_gap_report/gap_report.json]
- **route_efficiency**: missing (no routing_manifest.json found in output/)
- **algorithm_readiness**: available [robot_sf/benchmark/algorithm_readiness.py]
- **open_issues_snapshot**: available [docs/context/evidence/issue_2792_next_issue_shortlist/open_issues_snapshot.json]
- **recent_prs_snapshot**: available [docs/context/evidence/issue_2792_next_issue_shortlist/recent_prs_snapshot.json]

## Degradation Notes

- route_efficiency: no routing_manifest.json found in output/ -- signal skipped, no route-efficiency ranking component applied

## Candidates (92 total)

### #1 ledger-pedestrian_density_stress (score: 95)

- **Title**: Address gap in pedestrian_density_stress
- **Source**: dissertation_evidence_ledger
- **Source issue**: #1240
- **Bucket**: negative_revise_only
- **Evidence tier**: diagnostic
- **Reason**: Requires runtime execution evidence, failure-semantics classification, and planner-comparison rows before any pedestrian-density stress benchmark claim.
- **Blockers**: Coverage entropy is a diagnostic planning aid. It does not prove benchmark value, runtime stress effectiveness, or planner ranking. Dense stress rows should not be promoted without runtime and failure-semantics proof.
- **Data-source status**: available

### #2 ledger-observation_robustness (score: 85)

- **Title**: Address gap in observation_robustness
- **Source**: dissertation_evidence_ledger
- **Source issue**: #1246
- **Bucket**: negative_revise_only
- **Evidence tier**: release-backed
- **Reason**: Requires actual perception pipeline, calibrated tracking, or simulator-level observation fidelity before any robustness claim. Cross-track comparison is not valid without matched observation contracts.
- **Blockers**: The levels are benchmark evidence labels and compatibility gates, not sim-to-real validity claims. No real camera perception, detector training, calibrated tracking, or new environment observation implementation exists yet.
- **Data-source status**: available

### #3 gap-issue-2716-topology-reselection-cross-slice (score: 80)

- **Title**: Design a successor targeting actual clearance or terminal-outcome movement. Stop same-family selector reruns on the canonical slice.
- **Source**: dissertation_gap_report
- **Source issue**: #2716
- **Bucket**: negative_revise_only
- **Result classification**: revise
- **Reason**: All 9 hard progress-gated rows ended horizon_exhausted (159 deadlock steps, 0 collision). The mechanism activates and generalizes diagnostically but does not clear any hard slice at h160. Negative-control rows succeeded cleanly (3/3 success, 0 topology switches).
- **Blockers**: Diagnostic-only, not benchmark or paper evidence. Classification is revise, not promote.
- **Data-source status**: available

### #4 gap-issue-2749-observation-noise-distant-pedestrian (score: 80)

- **Title**: Retest with a closer pedestrian, higher pedestrian count, or a scenario where the planner must actively avoid pedestrians.
- **Source**: dissertation_gap_report
- **Source issue**: #2749
- **Bucket**: negative_revise_only
- **Result classification**: diagnostic_only
- **Reason**: All progress, risk, and planner metrics are identical between baseline and perception-limited. The single pedestrian is too far from the robot to influence planner decisions; the planner is dominated by static obstacle clearance. Perturbation plumbing is confirmed working (occlusion masking active, 1 missed detection at step 15) but no behavioral degradation is measurable.
- **Blockers**: Diagnostic-only, not benchmark or paper evidence. Confirms perturbation infrastructure works; does not measure behavioral degradation.
- **Data-source status**: available

### #5 gap-issue-2760-dissertation-evidence-ledger-diagnostic-rows (score: 80)

- **Title**: Use the ledger as a planning aid to identify which thesis areas need new experiments before Results wording is possible. Do not cite any current row as benchmark-strength evidence.
- **Source**: dissertation_gap_report
- **Source issue**: #2760
- **Bucket**: negative_revise_only
- **Result classification**: diagnostic_only
- **Reason**: Four of six areas are classified as diagnostic-tier evidence (topology, signalized behavior, prediction, pedestrian density). One area (observation robustness) is release-backed but is a contract/provenance layer only. One area (exported tables) is stale/non-claimable. No row supports Results-chapter wording without qualification.
- **Blockers**: Synthesis/planning aid only. Does not produce new benchmark evidence, paper-facing results, or safety claims.
- **Data-source status**: available

### #6 issue-2716-topology-reselection-cross-slice (score: 80)

- **Title**: Design a successor targeting actual clearance or terminal-outcome movement. Stop same-family selector reruns on the canonical slice.
- **Source**: negative_result_register
- **Source issue**: #2716
- **Bucket**: negative_revise_only
- **Result classification**: revise
- **Reason**: All 9 hard progress-gated rows ended horizon_exhausted (159 deadlock steps, 0 collision). The mechanism activates and generalizes diagnostically but does not clear any hard slice at h160. Negative-control rows succeeded cleanly (3/3 success, 0 topology switches).
- **Blockers**: Diagnostic-only, not benchmark or paper evidence. Classification is revise, not promote.
- **Data-source status**: available

### #7 issue-2749-observation-noise-distant-pedestrian (score: 80)

- **Title**: Retest with a closer pedestrian, higher pedestrian count, or a scenario where the planner must actively avoid pedestrians.
- **Source**: negative_result_register
- **Source issue**: #2749
- **Bucket**: negative_revise_only
- **Result classification**: diagnostic_only
- **Reason**: All progress, risk, and planner metrics are identical between baseline and perception-limited. The single pedestrian is too far from the robot to influence planner decisions; the planner is dominated by static obstacle clearance. Perturbation plumbing is confirmed working (occlusion masking active, 1 missed detection at step 15) but no behavioral degradation is measurable.
- **Blockers**: Diagnostic-only, not benchmark or paper evidence. Confirms perturbation infrastructure works; does not measure behavioral degradation.
- **Data-source status**: available

### #8 issue-2760-dissertation-evidence-ledger-diagnostic-rows (score: 80)

- **Title**: Use the ledger as a planning aid to identify which thesis areas need new experiments before Results wording is possible. Do not cite any current row as benchmark-strength evidence.
- **Source**: negative_result_register
- **Source issue**: #2760
- **Bucket**: negative_revise_only
- **Result classification**: diagnostic_only
- **Reason**: Four of six areas are classified as diagnostic-tier evidence (topology, signalized behavior, prediction, pedestrian density). One area (observation robustness) is release-backed but is a contract/provenance layer only. One area (exported tables) is stale/non-claimable. No row supports Results-chapter wording without qualification.
- **Blockers**: Synthesis/planning aid only. Does not produce new benchmark evidence, paper-facing results, or safety claims.
- **Data-source status**: available

### #9 gap-prediction (score: 75)

- **Title**: denominator repair: an executed planner campaign with matched metrics and fail-closed row handling is required before any benchmark claim.
- **Source**: dissertation_gap_report
- **Source issue**: #2475
- **Bucket**: blocked
- **Evidence tier**: diagnostic
- **Reason**: Requires executed planner campaign with reactive, single-trajectory, and multimodal rows; matched metrics (success, collision, min-ped-distance, time-to-goal); and fail-closed row handling before any benchmark claim.
- **Blockers**: This is contract evidence only. The runner uses deterministic fixtures, does not execute a planner campaign, and does not measure prediction quality or compare planning performance.
- **Data-source status**: available

### #10 gap-signalized_behavior (score: 75)

- **Title**: live replay: explicit runtime signal phase state, planner-observation policy, and planner_observable promotion are required before any benchmark row.
- **Source**: dissertation_gap_report
- **Source issue**: #2527
- **Bucket**: blocked
- **Evidence tier**: diagnostic
- **Reason**: Requires explicit runtime signal phase state, planner-observation policy, zone/legality trace fields, and planner_observable promotion before any benchmark row.
- **Blockers**: Signal state is proxy_diagnostic only. Do not claim planner observability, forced-waiting reasoning, legality compliance, or benchmark ranking improvement.
- **Data-source status**: available

### #11 gap-topology_guidance (score: 75)

- **Title**: live replay: a fresh topology-guided planner campaign with non-same-family selector variants and an independent seed set is required before any benchmark promotion.
- **Source**: dissertation_gap_report
- **Source issue**: #2518
- **Bucket**: blocked
- **Evidence tier**: diagnostic
- **Reason**: A new hypothesis with a different mechanism and metric is needed before further topology work. No current row supports Results wording.
- **Blockers**: Do not claim topology guidance improves success, transfer, or leaderboard performance. The lane is stop for same-family selector reruns on the canonical slice.
- **Data-source status**: available

### #12 ledger-prediction (score: 75)

- **Title**: denominator repair: an executed planner campaign with matched metrics and fail-closed row handling is required before any benchmark claim.
- **Source**: dissertation_evidence_ledger
- **Source issue**: #2475
- **Bucket**: blocked
- **Evidence tier**: diagnostic
- **Reason**: Requires executed planner campaign with reactive, single-trajectory, and multimodal rows; matched metrics (success, collision, min-ped-distance, time-to-goal); and fail-closed row handling before any benchmark claim.
- **Blockers**: This is contract evidence only. The runner uses deterministic fixtures, does not execute a planner campaign, and does not measure prediction quality or compare planning performance.
- **Data-source status**: available

### #13 ledger-signalized_behavior (score: 75)

- **Title**: live replay: explicit runtime signal phase state, planner-observation policy, and planner_observable promotion are required before any benchmark row.
- **Source**: dissertation_evidence_ledger
- **Source issue**: #2527
- **Bucket**: blocked
- **Evidence tier**: diagnostic
- **Reason**: Requires explicit runtime signal phase state, planner-observation policy, zone/legality trace fields, and planner_observable promotion before any benchmark row.
- **Blockers**: Signal state is proxy_diagnostic only. Do not claim planner observability, forced-waiting reasoning, legality compliance, or benchmark ranking improvement.
- **Data-source status**: available

### #14 ledger-topology_guidance (score: 75)

- **Title**: live replay: a fresh topology-guided planner campaign with non-same-family selector variants and an independent seed set is required before any benchmark promotion.
- **Source**: dissertation_evidence_ledger
- **Source issue**: #2518
- **Bucket**: blocked
- **Evidence tier**: diagnostic
- **Reason**: A new hypothesis with a different mechanism and metric is needed before further topology work. No current row supports Results wording.
- **Blockers**: Do not claim topology guidance improves success, transfer, or leaderboard performance. The lane is stop for same-family selector reruns on the canonical slice.
- **Data-source status**: available

### #15 ledger-exported_tables (score: 70)

- **Title**: stale artifact refresh: payload file recovery or re-export from a fresh campaign is required before any reuse.
- **Source**: dissertation_evidence_ledger
- **Source issue**: #1023
- **Bucket**: blocked
- **Evidence tier**: non-claimable
- **Reason**: Requires payload file recovery or re-export before any reuse.
- **Blockers**: The claim matrix reports missing payload files for both artifacts, and the stale-artifact detector classifies the bundle manifest as stale. The tables remain historical tracked evidence and do not establish new benchmark claims.
- **Data-source status**: available

### #16 algo-adaptive_proxemic_selector_v0 (score: 60)

- **Title**: Validate experimental algorithm: adaptive_proxemic_selector_v0
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Diagnostic selector over fixed conservative, neutral, and open proxemic hybrid-rule profile candidates.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #17 algo-adaptive_proxemic_selector_v1 (score: 60)

- **Title**: Validate experimental algorithm: adaptive_proxemic_selector_v1
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Diagnostic neutral-default selector over fixed proxemic hybrid-rule profiles; open profile is reserved for sparse low-progress recovery.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #18 algo-crowdnav_height (score: 60)

- **Title**: Validate experimental algorithm: crowdnav_height
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Upstream CrowdNav_HEIGHT model-only checkpoint wrapper.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #19 algo-dr_mpc (score: 60)

- **Title**: Validate experimental algorithm: dr_mpc
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: External DR-MPC wrapper; dependency-sensitive assessment anchor.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #20 algo-drl_vo (score: 60)

- **Title**: Validate experimental algorithm: drl_vo
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: DRL-VO hybrid planner (learned policy augmented with velocity obstacle fallback); prototype stage.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #21 algo-gap_prediction (score: 60)

- **Title**: Validate experimental algorithm: gap_prediction
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Predictive planner with stream-gap veto layer.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #22 algo-gensafenav_gst_predictor_rand (score: 60)

- **Title**: Validate experimental algorithm: gensafenav_gst_predictor_rand
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Upstream GenSafeNav CrowdNav++-style learned checkpoint wrapper with fail-fast asset checks.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #23 algo-gensafenav_gst_predictor_rand_guarded (score: 60)

- **Title**: Validate experimental algorithm: gensafenav_gst_predictor_rand_guarded
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: GenSafeNav GST_predictor_rand wrapper with explicit short-horizon safety guard and goal fallback for static-risk-heavy slices.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #24 algo-gensafenav_ours_gst (score: 60)

- **Title**: Validate experimental algorithm: gensafenav_ours_gst
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Upstream GenSafeNav constrained learned checkpoint wrapper with fail-fast asset checks.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #25 algo-gensafenav_ours_gst_guarded (score: 60)

- **Title**: Validate experimental algorithm: gensafenav_ours_gst_guarded
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: GenSafeNav Ours_GST wrapper with explicit short-horizon safety guard and goal fallback for static-risk-heavy slices.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #26 algo-grid_route (score: 60)

- **Title**: Validate experimental algorithm: grid_route
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Testing-only occupancy-grid route planner for static obstacle slices.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #27 algo-guarded_ppo (score: 60)

- **Title**: Validate experimental algorithm: guarded_ppo
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: PPO baseline with short-horizon safety veto and local fallback.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #28 algo-hrvo (score: 60)

- **Title**: Validate experimental algorithm: hrvo
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Local hybrid reciprocal velocity obstacles planner.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #29 algo-hybrid_orca_sampler (score: 60)

- **Title**: Validate experimental algorithm: hybrid_orca_sampler
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: ORCA primary planner with short-horizon MPPI repair for stalled or unsafe scenes.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #30 algo-hybrid_portfolio (score: 60)

- **Title**: Validate experimental algorithm: hybrid_portfolio
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Risk-regime switch between risk_dwa, ORCA, and prediction planner.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #31 algo-hybrid_rule_local_planner (score: 60)

- **Title**: Validate experimental algorithm: hybrid_rule_local_planner
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Deterministic hybrid-rule local planner family; v0 is minimal DWA-style. Actuation-aware aliases are synthetic diagnostic-only candidates.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #32 algo-lidar_grid_route (score: 60)

- **Title**: Validate experimental algorithm: lidar_grid_route
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Testing-only LiDAR-derived ego occupancy adapter wrapped around grid_route; not benchmark evidence without explicit opt-in.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #33 algo-lidar_social_force (score: 60)

- **Title**: Validate experimental algorithm: lidar_social_force
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Testing-only LiDAR endpoint-cluster tracked-agent adapter wrapped around SocialForce; not benchmark evidence without explicit opt-in.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #34 algo-mppi_social (score: 60)

- **Title**: Validate experimental algorithm: mppi_social
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Sampling-based MPPI/CEM social local planner.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #35 algo-nmpc_social (score: 60)

- **Title**: Validate experimental algorithm: nmpc_social
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Native NMPC-style local planner with short-horizon nonlinear optimization.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #36 algo-planner_selector_v2_diagnostic (score: 60)

- **Title**: Validate experimental algorithm: planner_selector_v2_diagnostic
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Diagnostic-only deterministic selector over existing local planner candidates; not benchmark-strength evidence.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #37 algo-policy_stack_v1 (score: 60)

- **Title**: Validate experimental algorithm: policy_stack_v1
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Minimal non-learning portfolio over goal and risk_dwa proposal sources.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #38 algo-ppo (score: 60)

- **Title**: Validate experimental algorithm: ppo
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Learned PPO baseline (paper profile requires provenance + quality gate).
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #39 algo-prediction_planner (score: 60)

- **Title**: Validate experimental algorithm: prediction_planner
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: RGL-inspired predictive planner; requires trained checkpoint.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #40 algo-predictive_mppi (score: 60)

- **Title**: Validate experimental algorithm: predictive_mppi
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Learned-prediction sequence optimizer over short action horizons.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #41 algo-risk_dwa (score: 60)

- **Title**: Validate experimental algorithm: risk_dwa
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Risk-aware dynamic-window planner (non-learning).
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #42 algo-risk_surface_dwa (score: 60)

- **Title**: Validate experimental algorithm: risk_surface_dwa
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Deterministic local risk-surface producer wrapped around risk_dwa; prototype-only and not learned-risk benchmark evidence.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #43 algo-sac (score: 60)

- **Title**: Validate experimental algorithm: sac
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Learned SB3 SAC baseline; benchmarkable only after checkpoint-specific quality gate.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #44 algo-sacadrl (score: 60)

- **Title**: Validate experimental algorithm: sacadrl
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: GA3C-CADRL adapter; dependency/model-sensitive.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #45 algo-safety_barrier (score: 60)

- **Title**: Validate experimental algorithm: safety_barrier
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Testing-only clean-room static-obstacle safety-barrier planner.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #46 algo-sicnav (score: 60)

- **Title**: Validate experimental algorithm: sicnav
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: External SICNav MPC wrapper; dependency-sensitive and testing-only.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #47 algo-social_navigation_pyenvs_hsfm_new_guo (score: 60)

- **Title**: Validate experimental algorithm: social_navigation_pyenvs_hsfm_new_guo
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Upstream Social-Navigation-PyEnvs non-trainable HSFM-New-Guo wrapper.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #48 algo-social_navigation_pyenvs_orca (score: 60)

- **Title**: Validate experimental algorithm: social_navigation_pyenvs_orca
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Upstream Social-Navigation-PyEnvs non-trainable ORCA wrapper.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #49 algo-social_navigation_pyenvs_sfm_helbing (score: 60)

- **Title**: Validate experimental algorithm: social_navigation_pyenvs_sfm_helbing
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Upstream Social-Navigation-PyEnvs non-trainable SFM-Helbing wrapper.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #50 algo-social_navigation_pyenvs_socialforce (score: 60)

- **Title**: Validate experimental algorithm: social_navigation_pyenvs_socialforce
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Upstream Social-Navigation-PyEnvs non-trainable SocialForce wrapper.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #51 algo-socnav_bench (score: 60)

- **Title**: Validate experimental algorithm: socnav_bench
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: SocNav benchmark adapter; dependency-sensitive.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #52 algo-socnav_hrvo (score: 60)

- **Title**: Validate experimental algorithm: socnav_hrvo
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: SocNav HRVO variant with hybrid reciprocal velocity obstacles.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #53 algo-socnav_orca_dd (score: 60)

- **Title**: Validate experimental algorithm: socnav_orca_dd
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: SocNav ORCA variant tuned for differential-drive compatibility.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #54 algo-socnav_orca_nonholonomic (score: 60)

- **Title**: Validate experimental algorithm: socnav_orca_nonholonomic
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: SocNav ORCA variant tuned for nonholonomic commitment.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #55 algo-socnav_orca_relaxed (score: 60)

- **Title**: Validate experimental algorithm: socnav_orca_relaxed
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: SocNav ORCA variant with relaxed safety tuning.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #56 algo-socnav_sampling (score: 60)

- **Title**: Validate experimental algorithm: socnav_sampling
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: SocNav sampling adapter; dependency-sensitive.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #57 algo-sonic_crowdnav (score: 60)

- **Title**: Validate experimental algorithm: sonic_crowdnav
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Upstream SoNIC model-only checkpoint wrapper with fail-fast source asset checks.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #58 algo-stream_gap (score: 60)

- **Title**: Validate experimental algorithm: stream_gap
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Gap-acceptance local planner for crossing/bottleneck experiments.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #59 algo-teb (score: 60)

- **Title**: Validate experimental algorithm: teb
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Native corridor-commitment planner inspired by TEB-style local optimization.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #60 algo-topology_guided_hybrid_rule_v0 (score: 60)

- **Title**: Validate experimental algorithm: topology_guided_hybrid_rule_v0
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Diagnostic-only masked-route hypothesis selector feeding the hybrid-rule local scorer; not benchmark evidence.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #61 algo-trivial_reference (score: 60)

- **Title**: Validate experimental algorithm: trivial_reference
- **Source**: algorithm_readiness
- **Bucket**: blocked
- **Reason**: Diagnostic starter-template adapter for contributor onboarding; not benchmark evidence.
- **Blockers**: Requires benchmark validation before baseline promotion
- **Data-source status**: available

### #62 open-issue-2414 (score: 60)

- **Title**: data: validate SocNavBench ETH traversible manifest and run eth_first converter preflight
- **Source**: open_issues_snapshot
- **Source issue**: #2414
- **Bucket**: blocked
- **Evidence tier**: blocked
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #63 open-issue-2415 (score: 60)

- **Title**: data: stage real AMV command-response trace manifest through amv-calibration path
- **Source**: open_issues_snapshot
- **Source issue**: #2415
- **Bucket**: blocked
- **Evidence tier**: blocked
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #64 open-issue-2416 (score: 60)

- **Title**: blocked: hold AMV profile implementation until source provenance is accepted
- **Source**: open_issues_snapshot
- **Source issue**: #2416
- **Bucket**: blocked
- **Evidence tier**: blocked
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #65 open-issue-2417 (score: 60)

- **Title**: blocked: keep AMV paper-facing unlock criteria gated on real source status
- **Source**: open_issues_snapshot
- **Source issue**: #2417
- **Bucket**: blocked
- **Evidence tier**: blocked
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #66 open-issue-2444 (score: 60)

- **Title**: analysis: find AMMV trace pair with nonzero mechanism divergence
- **Source**: open_issues_snapshot
- **Source issue**: #2444
- **Bucket**: blocked
- **Evidence tier**: blocked
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #67 open-issue-2445 (score: 60)

- **Title**: research: classify ORCA-residual progress-probe target after v1 smoke
- **Source**: open_issues_snapshot
- **Source issue**: #2445
- **Bucket**: blocked
- **Evidence tier**: blocked
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #68 open-issue-2446 (score: 60)

- **Title**: research: evaluate actuation feasibility as a diagnostic ranking dimension
- **Source**: open_issues_snapshot
- **Source issue**: #2446
- **Bucket**: blocked
- **Evidence tier**: analysis-only
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #69 open-issue-2546 (score: 60)

- **Title**: research: evaluate ScenarioBelief uncertainty effects on local policy observations
- **Source**: open_issues_snapshot
- **Source issue**: #2546
- **Bucket**: blocked
- **Evidence tier**: stress
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #70 open-issue-2601 (score: 60)

- **Title**: research: define adversarial manifest quality metrics after first planner smoke
- **Source**: open_issues_snapshot
- **Source issue**: #2601
- **Bucket**: blocked
- **Evidence tier**: proposal
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #71 open-issue-2722 (score: 60)

- **Title**: schema: add manifest lineage graph report
- **Source**: open_issues_snapshot
- **Source issue**: #2722
- **Bucket**: blocked
- **Evidence tier**: nominal
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #72 open-issue-2723 (score: 60)

- **Title**: schema: add manifest lineage migration and backfill validator
- **Source**: open_issues_snapshot
- **Source issue**: #2723
- **Bucket**: blocked
- **Evidence tier**: smoke
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #73 open-issue-2726 (score: 60)

- **Title**: data: calibrate scenario priors from benchmark trace clusters
- **Source**: open_issues_snapshot
- **Source issue**: #2726
- **Bucket**: blocked
- **Evidence tier**: analysis-only
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #74 open-issue-2727 (score: 60)

- **Title**: benchmark: add fast-moving bicycle actor as a trace-compatible dynamic obstacle
- **Source**: open_issues_snapshot
- **Source issue**: #2727
- **Bucket**: blocked
- **Evidence tier**: stress
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #75 open-issue-2754 (score: 60)

- **Title**: benchmark: add signalized-crossing failure-case pack with trace snapshots
- **Source**: open_issues_snapshot
- **Source issue**: #2754
- **Bucket**: blocked
- **Evidence tier**: analysis-only
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #76 open-issue-2758 (score: 60)

- **Title**: prediction: add signal-aware and goal-aware forecast baselines
- **Source**: open_issues_snapshot
- **Source issue**: #2758
- **Bucket**: blocked
- **Evidence tier**: nominal
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #77 open-issue-2759 (score: 60)

- **Title**: benchmark: connect forecast uncertainty to local-policy risk scoring
- **Source**: open_issues_snapshot
- **Source issue**: #2759
- **Bucket**: blocked
- **Evidence tier**: stress
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #78 open-issue-2761 (score: 60)

- **Title**: publication: add chapter-target labels to claim matrix rows
- **Source**: open_issues_snapshot
- **Source issue**: #2761
- **Bucket**: blocked
- **Evidence tier**: synthesis
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #79 open-issue-2763 (score: 60)

- **Title**: workflow: connect PR-loop dry-run policy to routed-worker artifact manifests
- **Source**: open_issues_snapshot
- **Source issue**: #2763
- **Bucket**: blocked
- **Evidence tier**: nominal
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #80 open-issue-2765 (score: 60)

- **Title**: benchmark: add dense pedestrian interaction stress slice for prediction and observation diagnostics
- **Source**: open_issues_snapshot
- **Source issue**: #2765
- **Bucket**: blocked
- **Evidence tier**: stress
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #81 open-issue-2766 (score: 60)

- **Title**: research: compare topology, signal, prediction, and observation evidence in one mixed-scenario matrix
- **Source**: open_issues_snapshot
- **Source issue**: #2766
- **Bucket**: blocked
- **Evidence tier**: synthesis
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #82 open-issue-2767 (score: 60)

- **Title**: paper: draft benchmark-results table candidates from claim matrix
- **Source**: open_issues_snapshot
- **Source issue**: #2767
- **Bucket**: blocked
- **Evidence tier**: synthesis
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #83 open-issue-2781 (score: 60)

- **Title**: prediction: add interaction-aware baseline before learned prediction
- **Source**: open_issues_snapshot
- **Source issue**: #2781
- **Bucket**: blocked
- **Evidence tier**: nominal
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #84 open-issue-2783 (score: 60)

- **Title**: publication: propagate evidence ledger and negative-result register into issue ranking
- **Source**: open_issues_snapshot
- **Source issue**: #2783
- **Bucket**: blocked
- **Evidence tier**: nominal
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #85 open-issue-2785 (score: 60)

- **Title**: research: use negative-result register to prevent repeated weak scenarios
- **Source**: open_issues_snapshot
- **Source issue**: #2785
- **Bucket**: blocked
- **Evidence tier**: nominal
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #86 open-issue-2786 (score: 60)

- **Title**: workflow: add historical route-efficiency dashboard
- **Source**: open_issues_snapshot
- **Source issue**: #2786
- **Bucket**: blocked
- **Evidence tier**: analysis-only
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #87 open-issue-2787 (score: 60)

- **Title**: workflow: add routing policy feedback from route-efficiency report
- **Source**: open_issues_snapshot
- **Source issue**: #2787
- **Bucket**: blocked
- **Evidence tier**: nominal
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #88 open-issue-2788 (score: 60)

- **Title**: adversarial: generate scenario candidates from negative-result register entries
- **Source**: open_issues_snapshot
- **Source issue**: #2788
- **Bucket**: blocked
- **Evidence tier**: smoke
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #89 open-issue-2790 (score: 60)

- **Title**: analysis: compare trace-derived and live-replay evidence after #2777
- **Source**: open_issues_snapshot
- **Source issue**: #2790
- **Bucket**: blocked
- **Evidence tier**: analysis-only
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #90 open-issue-2792 (score: 60)

- **Title**: planning: generate next-issue shortlist from open gaps and closed PR evidence
- **Source**: open_issues_snapshot
- **Source issue**: #2792
- **Bucket**: blocked
- **Evidence tier**: analysis-only
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #91 open-issue-2793 (score: 60)

- **Title**: benchmark: add bicycle forecast/evaluation exclusion contract
- **Source**: open_issues_snapshot
- **Source issue**: #2793
- **Bucket**: blocked
- **Evidence tier**: smoke
- **Reason**: Open issue snapshot candidate.
- **Data-source status**: available

### #92 gap-exported_tables (score: 50)

- **Title**: stale artifact refresh: payload file recovery or re-export from a fresh campaign is required before any reuse.
- **Source**: dissertation_gap_report
- **Source issue**: #1023
- **Bucket**: remove_weaken
- **Evidence tier**: non-claimable
- **Reason**: Requires payload file recovery or re-export before any reuse.
- **Blockers**: The claim matrix reports missing payload files for both artifacts, and the stale-artifact detector classifies the bundle manifest as stale. The tables remain historical tracked evidence and do not establish new benchmark claims.
- **Data-source status**: available

## Claim Boundaries

- This shortlist is a synthesis/planning aid. It does not produce new benchmark evidence, paper-facing results, or safety claims.
- Ranking is deterministic: same inputs always produce the same order.
- Missing sources are recorded as degradation notes; the shortlist still emits candidates from available sources.
- No GitHub mutation, no Project #5 writes, and no issue creation occurred.
