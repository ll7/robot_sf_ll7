# Context Notes Workflow

`docs/context/` is the repository's Markdown knowledge base for issue execution history, durable
agent handoff, and reusable reasoning that should not be trapped in chat or PR text.

Use this directory for non-trivial insights, decisions, tradeoffs, validation notes, and execution
context that future contributors or agents are likely to need again.

## When To Update An Existing Note

Prefer updating an existing note when:

* the same issue, planner family, workflow, or benchmark surface is already documented there, 
* the new work changes or clarifies an existing conclusion, 
* or splitting the note would make the decision trail harder to follow.

When you update a note, preserve the current source of truth and remove ambiguity:

* replace outdated statements when the old wording is no longer useful, 
* add dated outcome updates when historical context still matters, 
* and link to the validation commands, artifacts, or follow-up notes that justify the new state.

## When To Create A New Note

Create a new note when the subject is distinct enough that merging it into an existing document
would blur ownership or make the reasoning harder to locate.

Prefer these naming patterns:

* `issue_<number>_<topic>.md` for issue-scoped notes, 
* `<topic>_<date>.md` using `YYYY-MM-DD` for cross-issue audits, release notes, or bounded
  investigations.

## Required Linking

Every durable context note should link to the smallest useful set of related surfaces:

* the GitHub issue or PR that motivated the work, 
* canonical docs or configs that define the contract, 
* validation commands, artifacts, or output paths that support the conclusion, 
* predecessor or successor notes when a document is continued or superseded.

If the note changes repository guidance, also link it from a normal entry point such as
`docs/README.md` , `docs/dev_guide.md` , `AGENTS.md` , or `docs/ai/repo_overview.md` .

## Outdated And Superseded Content

Touched notes must not leave stale conclusions ambiguous.

If a note is still the canonical surface, update it in place.

If a note should remain for history but is no longer current, mark that clearly near the top:

```md
> Status: superseded by `docs/context/issue_999_new_note.md` on 2026-04-09.
> Keep this note only for historical context.
```

If the note is no longer useful even as history, remove the outdated statement instead of stacking
contradictory prose.

## Lightweight Structure

Use the smallest structure that keeps the note reusable. Most notes should include:

* the goal or decision, 
* the assumptions made and why they matter, 
* the key evidence or reasoning, 
* the validation path, 
* the current conclusion or follow-up boundary.

Avoid turning `docs/context/` into a scratchpad. Capture what future readers need to reuse the
knowledge, not every transient iteration detail.

## Skills And Entry Points

* Repository rule: [AGENTS.md](../../AGENTS.md)
* Contributor workflow: [docs/dev_guide.md](../dev_guide.md)
* Docs index entry: [docs/README.md](../README.md)
* AI-facing orientation: [docs/ai/repo_overview.md](../ai/repo_overview.md)
* Goal-driven agent loops:
  [goal_driven_agent_loops_2026-05-13.md](goal_driven_agent_loops_2026-05-13.md)
* Issue #1181/#1191 ML-Intern Bounded Assistant Assessment and Codex-Native Workflow Extraction:
  [issue_1181_ml_intern_experiment_assistant.md](issue_1181_ml_intern_experiment_assistant.md)
* Route-clearance certification: [issue_1105_route_clearance_certification.md](issue_1105_route_clearance_certification.md)
* Negative route-clearance repair: [issue_1130_negative_route_clearance_repair.md](issue_1130_negative_route_clearance_repair.md)
* Open-issues implementation status:
  [open_issues_implementation_status_2026-05-12.md](open_issues_implementation_status_2026-05-12.md)
* Open-issues maintainer-input triage:
  [open_issues_maintainer_input_triage.md](open_issues_maintainer_input_triage.md)
* Open-issues PR split strategy:
  [open_issues_pr_split_strategy_2026-05-13.md](open_issues_pr_split_strategy_2026-05-13.md)
* Thursday development review 2026-05-21:
  [thursday_development_review_2026-05-21.md](thursday_development_review_2026-05-21.md)
* Worktree training preservation audit 2026-05-25:
  [worktree_training_preservation_audit_2026-05-25.md](worktree_training_preservation_audit_2026-05-25.md)
* Issue #1240 scenario coverage entropy:
  [issue_1240_scenario_coverage_entropy.md](issue_1240_scenario_coverage_entropy.md)
* Issue #1167 predictive obstacle-feature pipeline:
  [issue_1167_predictive_obstacle_pipeline.md](issue_1167_predictive_obstacle_pipeline.md)
* Issue #1550 predictive same-seed row summary schema:
  [issue_1550_predictive_same_seed_row_summary_schema.md](issue_1550_predictive_same_seed_row_summary_schema.md)
* Issue #1504 ego-conditioned feature contract:
  [issue_1504_ego_feature_contract.md](issue_1504_ego_feature_contract.md)
* Issue #1543 predictive v2 negative audit:
  [issue_1543_predictive_v2_negative_audit.md](issue_1543_predictive_v2_negative_audit.md)
* Issue #1542 manuscript claim evidence map:
  [issue_1542_manuscript_claim_evidence_map.md](issue_1542_manuscript_claim_evidence_map.md)
* Issue #1530 optional planner preflight audit:
  [issue_1530_optional_preflight_audit.md](issue_1530_optional_preflight_audit.md)
* Issue #1348 capability-aware map catalog design:
  [issue_1348_capability_map_catalog_design.md](issue_1348_capability_map_catalog_design.md)
* Issue #1414 parser capability metadata:
  [issue_1414_parser_capability_metadata.md](issue_1414_parser_capability_metadata.md)
* Issue #1413 map catalog schema and sync checker:
  [issue_1413_map_catalog_schema_sync.md](issue_1413_map_catalog_schema_sync.md)
* Issue #1415 capability-aware map resolver:
  [issue_1415_capability_map_resolver.md](issue_1415_capability_map_resolver.md)
* Issue #1416 converted-map cache evaluation:
  [issue_1416_converted_map_cache_evaluation.md](issue_1416_converted_map_cache_evaluation.md)
* Issue #1246 graded observation levels:
  [issue_1246_observation_levels.md](issue_1246_observation_levels.md)
* Issue #1239 human-model transfer robustness:
  [issue_1239_human_model_transfer.md](issue_1239_human_model_transfer.md)
* Issue #1255 open-issue dependency graph:
  [issue_1255_open_issue_dependency_graph.md](issue_1255_open_issue_dependency_graph.md)
* Issue #1247 safety shield contract:
  [issue_1247_safety_shield_contract.md](issue_1247_safety_shield_contract.md)
* Issue #1287 force-gradient interpolation vectorization:
  [issue_1287_force_gradient_vectorization.md](issue_1287_force_gradient_vectorization.md)
* Issue #1286 SNQI bootstrap stability:
  [issue_1286_snqi_bootstrap_stability.md](issue_1286_snqi_bootstrap_stability.md)
* Issue #1285 TODO-docstring backlog ratchet:
  [issue_1285_docstring_todo_ratchet.md](issue_1285_docstring_todo_ratchet.md)
* Issue #1271 seed-sensitivity explorer:
  [issue_1271_seed_sensitivity_explorer.md](issue_1271_seed_sensitivity_explorer.md)
* Issue #1272 Safety-Oriented Validation And Falsification Strategy:
  [issue_1272_validation_falsification_strategy.md](issue_1272_validation_falsification_strategy.md)
* Issue #1432 Adaptive Test Strategy Claim Audit (2026-05-22):
  [adaptive_test_claim_audit_2026-05.md](adaptive_test_claim_audit_2026-05.md)
* Issue #1457 Adversarial Map And Start-State Generation Protocol (2026-05-23):
  [issue_1457_adversarial_generation_protocol.md](issue_1457_adversarial_generation_protocol.md)
* Issue #1500 Adversarial Campaign Manifest Freeze (2026-05-26):
  [issue_1500_adversarial_manifest.md](issue_1500_adversarial_manifest.md)
* Issue #1304 pedestrian config boundary:
  [issue_1304_pedestrian_config_boundary.md](issue_1304_pedestrian_config_boundary.md)
* Issue #1342 GH-Act Runtime Requirements:
  [issue_1342_gh_act_runtime_requirements.md](issue_1342_gh_act_runtime_requirements.md)
* Issue #1387 Tentabot-style value scorer spike:
  [issue_1387_tentabot_value_scorer_spike.md](issue_1387_tentabot_value_scorer_spike.md)
* Issue #1344 paired AMV primary protocol report:
  [issue_1344_paired_amv_protocol_report.md](issue_1344_paired_amv_protocol_report.md)
* SLURM issue batch status 2026-05-21:
  [slurm_issue_batch_status_2026-05-21.md](slurm_issue_batch_status_2026-05-21.md)
* Issue #1397 Oracle Imitation Launch Packet:
  [issue_1397_oracle_imitation_launch_packet.md](issue_1397_oracle_imitation_launch_packet.md)
* Issue #1353 Broader AMV Baseline Preflight:
  [issue_1353_broader_amv_preflight.md](issue_1353_broader_amv_preflight.md)
* Issue #1546 AMV actuation-envelope stress slice:
  [issue_1546_amv_actuation_envelope_stress_slice.md](issue_1546_amv_actuation_envelope_stress_slice.md)
* Issue #1556 synthetic AMV actuation stress slice:
  [issue_1556_amv_actuation_stress_slice.md](issue_1556_amv_actuation_stress_slice.md)
* Issue #1398 metric rollup reconciliation:
  [issue_1398_metric_rollup_reconciliation.md](issue_1398_metric_rollup_reconciliation.md)
* Issue #1396 Shielded PPO Repair Launch Packet:
  [issue_1396_shielded_ppo_launch_packet.md](issue_1396_shielded_ppo_launch_packet.md)
* Issue #1395 Learned Risk Model Launch Packet:
  [issue_1395_learned_risk_launch_packet.md](issue_1395_learned_risk_launch_packet.md)
* Issue #1294 seed-sensitivity perturbations:
  [issue_1294_seed_sensitivity_perturbations.md](issue_1294_seed_sensitivity_perturbations.md)
* Issue archetypes and evidence tiers:
  [issue_1512_issue_archetypes.md](issue_1512_issue_archetypes.md)
* Issue #1532 archetype metadata sync decision:
  [issue_1532_archetype_metadata_sync_decision.md](issue_1532_archetype_metadata_sync_decision.md)
* Artifact evidence vocabulary:
  [artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md)
* PR first-pass review audit:
  [pr_first_pass_review_audit_2026-05-14.md](pr_first_pass_review_audit_2026-05-14.md)
* Issue #1289 SNQI Method Alias Retirement:
  [issue_1289_snqi_method_alias_retirement.md](issue_1289_snqi_method_alias_retirement.md)
* Issue #1288 JSONL append optimization:
  [issue_1288_jsonl_orjson_append.md](issue_1288_jsonl_orjson_append.md)
* Issue #1308 Act Local Workflow Evaluation 2026-05-18:
  [issue_1308_act_local_workflow_evaluation.md](issue_1308_act_local_workflow_evaluation.md)
* Issue #1111 CARLA Setup-Only Smoke 2026-05-18:
  [issue_1111_carla_setup_smoke.md](issue_1111_carla_setup_smoke.md)
* Issue #1322 SocNavBench device-placement cleanup:
  [issue_1322_socnavbench_device_placement_cleanup.md](issue_1322_socnavbench_device_placement_cleanup.md)
* Issue #1323 SocNavBench personal-space velocity:
  [issue_1323_socnavbench_personal_space_velocity.md](issue_1323_socnavbench_personal_space_velocity.md)
* Issue #1319 task bundles:
  [issue_1319_task_bundles.md](issue_1319_task_bundles.md)
* Issue #1365 Social Graph Observation Adapter:
  [issue_1365_social_graph_observation_adapter.md](issue_1365_social_graph_observation_adapter.md)
* Issue #1369 SAGE MPC-Transfer Planner Reproducibility:
  [issue_1369_sage_mpc_transfer_assessment.md](issue_1369_sage_mpc_transfer_assessment.md)
* Issue #1361 Command-Lattice Corridor-Deadlock Assessment (2026-05-20):
  [issue_1361_motion_primitive_corridor_deadlock.md](issue_1361_motion_primitive_corridor_deadlock.md)
* Issue #1360 External TEB Reference Assessment:
  [issue_1360_external_teb_assessment.md](issue_1360_external_teb_assessment.md)
* Issue #769/#1364 DRL-VO Assessment and Privileged-State Audit:
  [issue_769_drl_vo_assessment.md](issue_769_drl_vo_assessment.md)
* Issue #1245 Benchmark Claim Artifact:
  [issue_1245_benchmark_claim.md](issue_1245_benchmark_claim.md)
* Issue #1169 CARLA Live T1 Oracle Replay:
  [issue_1169_carla_live_replay.md](issue_1169_carla_live_replay.md)
* Issue #1437 CARLA Robot Actor Spawn Failure:
  [issue_1437_carla_robot_spawn.md](issue_1437_carla_robot_spawn.md)
* Issue #1428 ORCA-residual training lineage:
  [issue_1428_orca_residual_lineage.md](issue_1428_orca_residual_lineage.md)
* Issue #1440 CARLA Robot Spawn Projection:
  [issue_1440_carla_spawn_projection.md](issue_1440_carla_spawn_projection.md)
* Issue #1467 CARLA Replay Metrics:
  [issue_1467_carla_replay_metrics.md](issue_1467_carla_replay_metrics.md)
* Issue #1442 CARLA Native Spawn Probe (2026-05-24):
  [issue_1442_carla_native_spawn_probe.md](issue_1442_carla_native_spawn_probe.md)
* Issue #1430 CARLA Live Replay Parity 2026-05-21:
  [issue_1430_carla_live_parity.md](issue_1430_carla_live_parity.md)
* Issue #1436 CI Reproducibility and Flaky Statistical Acceptance Policy (2026-05-22):
  [issue_1436_reproducibility_flaky_acceptance.md](issue_1436_reproducibility_flaky_acceptance.md)
* Issue #1363 Learned Local Policy Eligibility Checklist:
  [policy_search/contracts/learned_local_policy_eligibility.md](policy_search/contracts/learned_local_policy_eligibility.md)
* Note-maintenance skill:
  [.agents/skills/context-note-maintainer/SKILL.md](../../.agents/skills/context-note-maintainer/SKILL.md)

## Example

* [docs/context/issue_796_agent_knowledge_capture_policy.md](issue_796_agent_knowledge_capture_policy.md)
* [docs/context/issue_805_teb_corridor_commitment_iteration.md](issue_805_teb_corridor_commitment_iteration.md)

## Evidence Bundles

* [Evidence Bundles](evidence/README.md) documents the narrow policy for promoting small generated
  artifacts out of `output/` into git. Current bundles include the
  [h500 policy-search evidence](evidence/policy_search_h500_2026-05-06/README.md),
  [issue 1023 candidate-augmented preflight](evidence/issue_1023_candidate_augmented_preflight_2026-05-06/README.md),
  [issue 1023 candidate-augmented local full campaign](evidence/issue_1023_candidate_augmented_local_full_2026-05-06/README.md),
  [issue 1023 scenario-horizon preflight](evidence/issue_1023_scenario_horizons_preflight_2026-05-06/README.md),
  [issue 1023 local full campaign](evidence/issue_1023_scenario_horizons_local_full_2026-05-06/README.md),
  [issue 1045 h500 solvability mechanisms](evidence/issue_1045_h500_solvability_mechanisms_2026-05-07/README.md),
  [issue 1462 S10/h500 failure modes](evidence/issue_1462_s10_h500_failure_modes_2026-05-24/README.md),
  [issue 1396 shielded PPO launch packet](evidence/issue_1396_shielded_ppo_launch_packet_2026-05-24/README.md),
  [issue 1395 learned-risk launch packet](evidence/issue_1395_learned_risk_launch_packet_2026-05-24/README.md),
  [issue 1397 oracle imitation launch packet](evidence/issue_1397_oracle_imitation_launch_packet_2026-05-24/README.md),
  [issue 1113 continuous h500 promotion evidence](evidence/issue_1113_continuous_h500_2026-05-10/README.md),
  [Issue #1111 CARLA Setup-Only Smoke Evidence](evidence/issue_1111_carla_setup_smoke_2026-05-18/README.md),
  [Issue #1239 Human-Model Transfer Evidence](evidence/issue_1239_human_model_transfer_2026-05-18/README.md),
  [Issue #1169 CARLA Live Replay Evidence](evidence/issue_1169_carla_live_replay_2026-05-18/README.md),
  [Issue #1467 CARLA Replay Metrics Evidence](evidence/issue_1467_carla_replay_metrics_2026-05-24/README.md),
  [Issue #1442 CARLA Native Spawn Probe Evidence](evidence/issue_1442_carla_native_spawn_probe_2026-05-24/README.md),
  [Issue #1344 Paired AMV Primary Evidence](evidence/issue_1344_paired_amv_primary_2026-05-20/README.md),
  [Issue #1416 Converted-Map Cache Evidence](evidence/issue_1416_converted_map_cache_2026-05-20/README.md),
  and the
  [May 4 camera-ready all-planners evidence](evidence/camera_ready_all_planners_2026-05-04/README.md).

## Benchmark Run Notes

* [Issue #1434 Stress/Uncertainty Coverage Schema v1](issue_1434_stress_uncertainty_coverage_schema.md)
  defines the `stress_uncertainty_coverage.v1` field contract, statistical summary tiers,
  scenario-parameter and failure-mode coverage axes, interpretation boundaries, and fail-closed
  consumer rules for benchmark reports. Implementation follow-up is issue #1445.
* [Issue #1454 S10 Robustness Campaign Plan](issue_1454_s10_robustness_plan.md)
  records the two-stage S10 fixed-h100 vs scenario-horizons-h500 robustness configs, contract
  tests, Stage A gate, and sensitivity-only output classification.
* [Issue #1454 Stage A Fixed-H100 Gate](issue_1454_stage_a_gate_2026-05-22.md)
  records the full S10 fixed-h100 campaign result, `socnav_bench` fail-closed blocker, analyzer
  output, May 4 comparison, and Stage B no-go decision.
* [Issue #1454 S10 H500 Candidate Comparison](issue_1454_s10_h500_candidate_comparison.md)
  records the exploratory h500 candidate comparison against previous h500, fixed-h100, and
  policy-search evidence, with the direct-metric claim boundary and SNQI caveat.
* [Issue #1545 Power-Aware Seed-Budget Planning](issue_1545_power_aware_seed_budget_planning.md)
  defines conservative smoke, nominal-sanity, compact-benchmark, and paper-facing seed tiers from
  existing durable summaries without changing benchmark gates or promoting post-hoc significance
  claims.
* [Camera-Ready All-Planners SLURM Check (2026-05-04)](camera_ready_all_planners_slurm_2026-05-04.md)
  records the failed `rsf-allbench` SLURM job, partial seven-planner campaign evidence,
  `socnav_bench` asset blocker, and rerun decision boundary for the May 4 all-planners matrix.
* [Issue #1023 Scenario-Horizon Benchmark Surface](issue_1023_scenario_horizon_benchmark.md)
  records the runnable paper-facing scenario-horizon config, preflight evidence, local non-Slurm
  full campaign, candidate-augmented local full campaign, fixed-vs-scenario comparison, and
  promotion boundary.
* [Issue #1318 TEB Corridor-Deadlock Evaluation](issue_1318_teb_corridor_deadlock_eval.md)
  records the tracked classic-merging corridor-deadlock slice, exact-collision metric integrity
  fix, TEB/ORCA/hybrid-rule comparison, and compact evidence bundle.
* [Issue #1081 Observation Noise](issue_1081_observation_noise.md) records the opt-in benchmark
  observation-noise profile contract, provenance fields, resume identity behavior, and
  non-calibrated robustness interpretation limit.
* [Issue #1038 H500 SNQI Contract Decision](issue_1038_h500_snqi_contract.md)
  records the h500 scenario-horizon SNQI failure decomposition and the decision to keep the surface
  experimental instead of overwriting the camera-ready v3 SNQI assets.
* [Issue #1045 H500 Solvability Mechanisms](issue_1045_h500_solvability_mechanisms.md)
  classifies aggregate fixed-to-h500 timeout relief into clean budget relief, late completion,
  exposure-enabled completion, partial relief, and safety-regressed completion while marking
  wait-then-go explanations as trace-required.
* [Issue #1462 S10 H500 Failure Modes](issue_1462_s10_h500_failure_modes.md)
  derives scenario, candidate-vs-core, and seed-level failure-mode tables from the issue #1454
  S10/h500 candidate evidence.
* [Issue #1051 Camera-Ready Evidence Provenance Audit](issue_1051_camera_ready_evidence_provenance_audit.md)
  inventories PPO provenance, release inputs, SNQI assets, seed/bootstrap evidence, and durable
  artifact gaps before the paper evidence trail is treated as archive-complete.
* [Issue #1053 Durable Artifact Reference Audit](issue_1053_durable_artifact_references.md)
  separates durable W&B/release/artifact-store pointers from local `output/` caches and records the
  publication-bundle archive blocker.
* [Issue #1062 Paper Evidence Archive Pointer](issue_1062_paper_evidence_archive.md)
  records the durable scoped `0.0.2` release archive, DOI, checksums, publication manifest, and SNQI
  diagnostics recovery path without committing raw benchmark outputs.
* [Issue #1054 Planner Readiness And Fallback Audit](issue_1054_planner_readiness_fallback_audit.md)
  records the paper-matrix planner readiness table, dependency status, and fail-closed treatment of
  fallback or degraded rows.
* [Issue #1049 H500 Mechanism Pilot](issue_1049_h500_mechanism_pilot.md)
  records compact fixed-h100 versus h500 ORCA traces for clean budget relief, exposure/comfort
  pressure, and safety-regressed long-horizon behavior.
* [Issue #1056 H500 Failure Classification](issue_1056_h500_failure_classification.md)
  defines the reusable h500 classification vocabulary and routes observed mechanisms to reporting,
  planner, or scenario-certification follow-up boundaries.
* [Issue #1055 Exposure-Aware H500 Tables](issue_1055_exposure_aware_h500_tables.md)
  defines and populates representative h500 reporting tables that place completion gains beside
  duration, collision, near-miss, force-exposure, and comfort-exposure rates.
* [Issue #1059 Deferred Planner-Improvement Program](issue_1059_deferred_planner_improvement_program.md)
  links the #1049/#1056 trace evidence to the first targeted planner child (#1034/#1036), records
  the strict h500 incumbent envelope, and keeps full-matrix promotion deferred to #1113.
* [Issue #1073 Robot SF Empirical-Expansion Gate](issue_1073_empirical_expansion_gate_2026_06_08.md)
  defines the June 8 checkpoint rule for promoting Robot SF beyond dissertation-floor examples,
  including counted improvement units, proof surfaces, and decline criteria.
* [Issue #1082 Paper Cross-Kinematics Parity Sweep](issue_1082_paper_cross_kinematics_v1.md)
  adds the `paper-cross-kinematics-v1` profile, three-mode kinematics matrix, compatibility
  manifest, and smoke/preflight command boundary for cross-kinematics interpretation.
* [Issue #1274 General Cross-Kinematics Parity Sweep](issue_1274_cross_kinematics_v1.md)
  adds the non-paper `cross_kinematics_v1` profile, compact scenario surface, compatibility
  manifest, and interpretation boundary for local parity checks.
* [Issue #1057 Semantic Blocker Audit](issue_1057_semantic_blocker_audit.md)
  classifies route handoff, invalid SVG repair, SNQI drift, metric sensitivity, fallback/degraded
  status, and live route-clearance warnings before planner-failure attribution.
* [Issue #1065 Route-Clearance Warning Audit](issue_1065_route_clearance_warning_audit.md)
  lists all current paper and h500 route-clearance warnings, classifies their attribution boundary,
  and opens the route repair/certification follow-up.
* [Issue #1058 H500 Paper Language](issue_1058_h500_paper_language.md)
  provides reusable paper/report wording for h500 as a long-horizon sensitivity surface and marks
  unsafe winner-table, wait-then-go, and SNQI calibration claims.
* [Issue #1085 Pedestrian-Impact Aggregate Metrics](issue_1085_pedestrian_impact_metrics.md)
  defines the schema-backed `pedestrian-impact.v1` block, canonical aggregate reductions, and
  opt-in CLI path for pedestrian-impact benchmark outputs.
* [Issue #1092 Multi-AMV First Slice](issue_1092_multi_amv_first_slice.md)
  records the minimal multi-robot scenario surface, smoke runner, inter-robot metrics, and deferred
  fleet-integration boundary.
* [Issue #1128 Multi-AMV Episode Extension](issue_1128_multi_amv_episode_extension.md)
  records the canonical `metrics.inter_robot` JSONL/report output contract for the explicit
  multi-AMV smoke path.
* [Issue #1168 Multi-AMV Planner Support Classification](issue_1168_multi_amv_planner_support.md)
  records the current planner-family inventory, fail-closed support gate, and the boundary between
  goal-controller smoke execution and real multi-AMV planner support.
* [Issue #1091 SDD Importer](issue_1091_sdd_importer.md)
  records the one-dataset-first real-world trajectory import boundary, SDD license assumptions,
  importer outputs, and deferred generalization scope.
* [Issue #1090 Observation Visibility](issue_1090_observation_visibility.md)
  records the planner-facing FOV/range/static-occlusion boundary, ground-truth separation, and
  dynamic-occlusion follow-up boundary.
* [Issue #1108 BC Warm-Start PPO Execution](issue_1108_bc_warm_start_execution.md)
  records the imitation observation-contract blocker, unblock patch, one-episode real collection
  preflight, and Slurm job IDs for the #749 BC-preinitialized PPO chain.
* [Issue #1083 Sanity V1 Nominal Matrix](issue_1083_sanity_v1_nominal_matrix.md)
  records the non-paper-facing nominal calibration matrix, smoke config, baseline threshold, and
  local proof run for easier deployment-like scenes.
* [Issue #1084 Planner Inclusion Gate](issue_1084_planner_inclusion_gate.md)
  records the mechanical planner inclusion-check command, report schema, default thresholds, and
  real pass/revise proof cases for promotion review.
* [Issue #1044 H500 Follow-Up Benchmark Plan](issue_1044_h500_followup_benchmark_plan.md)
  defines the long-horizon claim boundary, multi-table reporting plan, raw evidence requirements,
  pilot trace slice, and separate SNQI contract policy for a future h500 paper or benchmark report.
* [Issue #1052 Claim-Language Audit](issue_1052_claim_language_audit.md)
  records the benchmark-set claim boundary for issue-791 paper wording and marks stale OOD /
  transfer language as historical rather than current manuscript scope.
* [Issue #1074 Robot-SF Worked-Example Pack](issue_1074_robot_sf_worked_example_pack.md)
  curates three retained h500 examples that map scenario class, actor mix, metric layer,
  failure-pattern vocabulary, durable evidence, and claim/non-claim boundaries.
* [Issue #1075 Operating Envelope And Non-Claims](issue_1075_operating_envelope.md)
  defines the current Robot-SF dissertation-floor evidence envelope, supported evidence types,
  non-claims, and future-work boundaries for CARLA, physical validation, and broader empirical use.
* [Issue #1023 Experimental Benchmark Candidates](issue_1023_experimental_benchmark_candidates.md)
  records why `scenario_adaptive_hybrid_orca_v1` and
  `hybrid_rule_v3_fast_progress_static_escape` were added to the long-horizon benchmark as
  experimental candidates, plus their planner behavior and caveats.
* [Issue #1113 Continuous H500 Promotion Matrix](issue_1113_continuous_h500_promotion.md)
  records the full `full_matrix_h500` run for
  `hybrid_rule_v3_fast_progress_static_escape_continuous`, promotion-gate outcome, comparator
  deltas, remaining failure taxonomy, and artifact persistence boundary.
* [Issue #1152 Manual-Control Mode Experiments](issue_1152_manual_control_modes.md)
  records the first post-MVP steering-mode bundle, artifact-filterability contract, and fail-closed
  `ego_up_view_v1` renderer-hook blocker.

## Manual Control Notes

* [Issue #1163 Manual-Control Recording Format Decision](issue_1163_manual_control_recording_format.md)
  records the measured no-change decision for compact manual-control recording formats, including
  JSONL size/throughput thresholds and provenance requirements for any future derived artifact.
* [Issue #1154 Web-Game Data Collection Path](issue_1154_web_game_data_collection_path.md)
  records the feasibility gate for browser/web-game manual-control collection, including schema
  parity, consent/privacy, retention, deterministic hosted scenarios, and the narrow offline
  compatibility-prototype boundary.
* [Issue #1162 Manual-Control Active-Attempt Rewind](issue_1162_manual_control_rewind.md)
  records the replay-to-step rewind strategy, rejected simulator-checkpoint alternative,
  append-only metadata boundary, and fail-closed repeated-rewind policy.

## Feature Extractor Notes

* [Issue #193 Feature Extractor Evaluation](./issue_193_feature_extractor_evaluation.md)
  GPU throughput microbenchmark + 32 K PPO comparison of DynamicsExtractor vs MLP/CNN/Attention;
  preserves the historical `mlp_small` recommendation but marks default promotion as superseded
  pending the later `dyn_large_med` safety blocker and #834 maintainer decision.
* [Issue #193 Feature Extractor Optuna Study](./issue_193_feature_extractor_optuna_study.md)
  4 M-step SLURM sweep infrastructure, DB classification, and April 20 final pre-screen analysis; 
`feat_sweep_4m_array.db` is the current evidence surface, with longer 10 M+ validation still
  required before promotion.
* [Issue #835 Lightweight CNN Divergence Triage](./issue_835_lightweight_cnn_divergence.md)
  bounded 32 K rerun with PPO gradient and feature diagnostics; the issue-193 catastrophic
`lightweight_cnn` final drop did not reproduce, so the extractor remains experimental without an
  immediate architecture change.
* [Issue #850 PPO Collision Failures](./issue_850_ppo_collision_failures.md)
  follow-up diagnostics for the issue-193 `dyn_large_med` hold-out collision failures and the
  config-first safety-reward mitigation candidate.
* [Issue #863 SVG/Model Log Spam](./issue_863_svg_model_log_spam.md)
  log dedupe and PPO evaluation phase-marker decision for issue-791 long-run triage.
* [Issue #1037 RL Environment Patterns](./issue_1037_rl_environment_patterns.md)
  maps the May 2026 Hugging Face RL environment guide to Robot SF training, reward,
  rollout, benchmark, scaling, and provenance boundaries.
* [Issue #1291 PedestrianEnv Consolidation](./issue_1291_pedestrian_env_consolidation.md)
  records the removal of the transition-only `pedestrian_env_refactored.py` module and
  the compatibility aliases retained in the canonical `pedestrian_env.py` implementation.

## Training Notes

* [Issue #749 BC-Preinitialized PPO Launch Packet](issue_749_bc_preinit_ppo_launch_packet.md)
  defines the config-first launch path and artifact boundary for the deferred BC warm-start PPO
  challenger experiment.
* [Issue #1209 Imitation Observation Contract](issue_1209_imitation_observation_contract.md)
  records the BR-06 checkpoint-compatible observation-contract fix and validation path that
  unblocks #1108's BC warm-start launch.
* [Issue #1024 H500 PPO Retrain](issue_1024_h500_ppo_retrain.md)
  records the all-available scenario surface, PR #1025 h500 horizon alignment, and SLURM job
  `12350` for the first 12M-step PPO retrain.

## Performance Notes

* [Issue #1001 Architecture Seam Audit](./issue_1001_architecture_seam_audit.md)
  records benchmark/planner/training hotspot ownership boundaries, the top three refactor
  candidates, and the first no-behavior-change map-runner command-contract extraction.
* [Issue #1002 Complexity and Test Runtime Baseline](./issue_1002_complexity_runtime_baseline.md)
  adds a lightweight `scripts/dev/complexity_runtime_baseline.py` command and records the first
  2026-05-05 refactor-prioritization snapshot.
* [Issue #1006 GitHub CI Runtime Drift Diagnosis](./issue_1006_ci_runtime_drift.md)
  adds `scripts/dev/ci_timing_summary.py` and records timing evidence from PRs #1007 and #1008.
* [Issue #1290 Lazy Pygame Initialization](./issue_1290_lazy_pygame_init.md)
  records the lazy `SimulationView` proxy and pygame-free headless `make_robot_env(debug=False)`
  import contract, plus the cold/warm smoke caveat that backend/JIT startup still dominates.
* [Issue #513 High-Density Perf Gate Calibration](./issue_513_high_density_perf_gate.md)
  keeps `classic_cross_trap_high` advisory because no stable local trend-history window was
  available; documents the rerun evidence and non-blocking policy.
* [Issue #867 PPO Evaluation Reload Profile](./issue_867_ppo_eval_reload_profile.md)
  measurement-only issue-791 evaluation probe showing cached predictive-model reloads are small
  compared with shared cold startup and first-step overhead.
* [Issue #815 SAC Cold/Warm Performance Profile](./issue_815_sac_perf_cold_warm.md)
  cold/warm harness evidence showing the remaining issue-815 SAC simulator cost is localized to
  cold startup and lazy first-step initialization, not warm steady-state stepping.

## Planner Integration Notes

* [External Planner Reuse Checklist](./external_planner_reuse_checklist.md)
* [Issue #562 SocNavBench Re-Entry Gate](./issue_562_socnav_bench_reentry.md)
  defines the fail-fast focused probe and concrete keep-out/re-entry criteria for `socnav_bench`

  before it can be reconsidered for paper-facing benchmarks.
* [Issue #792 Social-Jym Source-Harness Reproduction](./issue_792_social_jym_source_harness.md)
  resolves the prior SSH-submodule blocker via HTTPS in an isolated side environment and proves a
  minimal upstream `SocialNav` reset plus `SARL` policy step, without claiming Robot SF benchmark
  integration.
* [Issue #905 Social-Jym Wrapper Spike](./issue_905_social_jym_wrapper_spike.md)
  proves a one-step Robot SF smoke loop through a thin random-action upstream `SARL` wrapper and
  records the remaining parity/provenance boundary.
* [Issue #907 Social-Jym SARL Wrapper Parity](./issue_907_social_jym_sarl_parity.md)
  proves controlled one-human SARL input parity for the wrapper and quantifies holonomic-to-unicycle
  projection loss, while keeping benchmark integration blocked on trained-policy provenance.
* [Issue #909 Social-Jym Trained-Policy Provenance](./issue_909_social_jym_policy_provenance.md)
  records the fail-closed finding that the pinned upstream checkout references SARL/SARL-PPO
  artifacts but does not include reproducible trained policy weights, so benchmark smoke remains
  unjustified.
* [Issue #626 SoNIC Source Harness Probe](./issue_626_sonic_source_harness_probe.md)
* [Issue #627 SoNIC Wrapper Follow-up](./issue_627_sonic_wrapper_followup.md)
* [Issue #1368 NeuPAN Point-Obstacle Comparator Assessment (2026-05-20)](issue_1368_neupan_point_obstacle_assessment.md)
* [Issue #1367 CrowdNav-Family Learned-Policy Verdict](./policy_search/issue_1367_crowdnav_family_verdict.md)
* [Issue #1394 CrowdNav HEIGHT Source-Harness Proof](./policy_search/issue_1394_crowdnav_height_source_harness.md)
* [Issue #1366 GenSafeNav / SoNIC Conformal Contract](./policy_search/issue_1366_gensafenav_sonic_conformal_contract.md)
* [Issue #1393 GenSafeNav / SoNIC Source-Harness Reproduction (2026-05-20)](./policy_search/issue_1393_gensafenav_source_harness.md)
* [Policy Search Context](./policy_search/README.md) - file-based candidate registry, staged local
  evaluation funnel, SLURM handoff notes, and the current
  [portfolio overview](./policy_search/portfolio_overview_2026-05-05.md) for the non-training
  policy-search workstream.
* [Issue #1443 Oracle Imitation Dataset Split Policy](./policy_search/contracts/oracle_imitation_dataset_split.md)
  defines the train/validation/evaluation seed-split contract, hard-slice assignment rules,
  relabeling boundaries, and manifest schema required before any oracle-imitation dataset
  generation begins.
* [Issue #1357 Tentabot-Style Motion-Primitive Assessment](./policy_search/2026-05-20_tentabot_motion_primitive_assessment.md)
  records the source-backed verdict that Tentabot-style learned primitive-value scoring is a
  Robot SF-native spike candidate, not an upstream adapter or benchmark-ready planner.
* [Issue #926 Policy Stack V1 Contract](issue_926_policy_stack_v1_contract.md)
  defines the minimal `policy_stack_v1` portfolio-planner contract, diagnostics, and benchmark
  claim boundary before runtime implementation under #871.
* [Issue #1004 Policy Stack V1 Runtime](issue_1004_policy_stack_v1_runtime.md)
  records the first runnable `policy_stack_v1` slice, its explicit proposal-status diagnostics, and
  the map-runner smoke limitation under parent #871.
* [Issue #871 Policy Stack Proposal Normalization](issue_871_policy_stack_proposal_normalization.md)
  hardens `policy_stack_v1` so malformed, non-finite, or command-bounds-violating child proposals
  are rejected before risk scoring.
* [Issue #871 Policy Stack Topology Smoke](issue_871_policy_stack_topology_smoke.md)
  records the `corridor_following` atomic topology smoke that proves `policy_stack_v1` can reach a
  topology-heavy goal through the normal map-runner path with proposal diagnostics intact.
* [Issue #884 Classic Merging Diagnostics](issue_884_classic_merging_diagnostics.md)
  records source-level hybrid-rule rejection diagnostics, rejected classic-merging recovery
  mechanisms, and the later #1034 targeted recovery result.
* [Issue #1027 Route-Corridor Attribution Diagnostics](issue_1027_route_corridor_attribution.md)
  records additive route-corridor `last_decision()` diagnostics, five regenerated #884 traces, and
  the geometry/dropout evidence needed before corridor-subgoal behavior work.
* [Issue #1028 Corridor-Subgoal Recovery](issue_1028_corridor_subgoal_recovery.md)
  records the disabled-by-default `corridor_subgoal` implementation, rejected enablement probes,
  and final #1029 h500 validation showing no target collision regressions but no #884 recovery.
* [Issue #1034 Continuous Corridor Maneuver](issue_1034_continuous_corridor_maneuver.md)
  records the environment-bound continuous static checks, rollout-sequence corridor maneuver,
  tracked candidate config, and h500 target/nominal/stress evidence for the #884 follow-up.
* [Issue #1022 Route-Corridor Design Research](issue_1022_route_corridor_design_research.md)
  records the research-first #884 follow-up: regenerated five-seed evidence, missing
  route-corridor diagnostics, design options, and the recommended diagnostic-first split.
  consolidates the #884 issue comments, rejected classic-merging recovery attempts, route-corridor
  research hypothesis, diagnostic contract, proof boundary, and research-first deferral through
  follow-up issue #1022.

## Map Coverage Notes

* [Issue #435 Map Coverage Flow](./issue_435_map_coverage_flow.md)
  parent flow state for real-world maps, SocNavBench import, and map-quality repair issues.
* [Issue #334 SocNavBench ETH Import Batch](issue_334_socnavbench_eth_import.md)
  records the first staged SocNavBench map-import batch, the fail-closed source-asset validator,
  and the boundary before a converted ETH SVG can be committed.
* [Issue #328 Real-World Map Parent Tracker](./issue_328_real_world_map_parent.md)
  parent/child split, current child issue state, and shared validation contract for real-world
  benchmark maps.

## Reasoning Notes

Design and decision rationale notes live in `docs/context/reasoning/` when the goal is to preserve
why a change was made rather than a full issue execution transcript.

* [Issue #592 Hybrid Obstacle-Context Predictor Design](./issue_592_hybrid_obstacle_predictor_design.md)
  scopes the obstacle-conditioned predictive-model idea into a feature-baseline-first experiment
  milestone that links follow-up #1138 for the first deterministic obstacle-feature
  implementation slice and defines a path with proof gates before any grid/CNN or obstacle-node
  graph prototype.
* [Issue #1165 Predictive Obstacle-Feature Lifecycle](issue_1165_predictive_obstacle_lifecycle.md)
  records the schema/dimension contract for obstacle-feature predictive datasets, training,
  checkpoints, runtime loading, and the #1218 map-derived obstacle-line data source before any
  same-seed performance comparison.
* [Issue #932 Hybrid Portfolio Diagnostics](./issue_932_hybrid_portfolio_diagnostics.md)
  records the first small policy-stack runtime diagnostics slice: selected-head counts, fallback
  counts, and last-decision metadata on `HybridPortfolioAdapter` .
* [Issue #938 Hybrid Portfolio Last Decision](./issue_938_hybrid_portfolio_last_decision.md)
  adds a step-level `HybridPortfolioAdapter.last_decision()` accessor consistent with nearby
  planner diagnostics APIs.
* [Issue #589 Public Leaderboard MVP Boundary](./issue_589_public_leaderboard_mvp.md)
  records the no-implementation-now decision, future PR-based MVP boundary, and prerequisites for
  any public planner leaderboard work.

## Execution Workflow Notes

* [SLURM Multi-Worktree Branch Workflow](slurm_multi_worktree_branch_workflow.md) - branch-isolated
  SLURM submissions from a shared login node, including `local.machine.md` symlink guidance and
  virtualenv boundaries.
* [Issue #1544 Slurm Experiment State Ledger](issue_1544_slurm_experiment_state_ledger.md) -
  lightweight state block, stale-trail closure protocol, local-vs-durable evidence boundary, and a
  conservative #1108 classification for execution issues.
* [Issue #856 PPO All-Scenarios Full Budget](issue_856_ppo_all_scenarios_full_budget.md) -
  broad-training PPO campaign record, camera-ready comparison, replica gate, and the horizon-500
  best-checkpoint Slurm handoff after the local env22 OOM.
* [Issue #845 Slurm Utilization Probe](issue_845_slurm_utilization_probe.md) - `sstat`/`sacct`/`seff`
  evidence collection path for diagnosing low CPU utilization without launching new jobs.
* [Issue #869 Adversarial Runner](issue_869_adversarial_runner.md) - programmable adversarial
  scenario search API, bundle contract, certification boundary, and deferred optimizer scope.
* [Issue #1237 Adversarial Failure Archive](issue_1237_adversarial_failure_archive.md) -
  compact `adversarial_failure_archive.v1` manifests for deterministic failure grouping and
  replay pointers without copying raw bundles.
* [Issue #1433 Adversarial Edge-Case Search Design (2026-05-22)](issue_1433_adversarial_edge_case_search_design.md) -
  bounded v1 design for crossing/TTC adversarial search: parameter bounds, invalid-candidate
  handling, scripted vs learned decisions, execution contract, failure classes, artifact policy,
  and explicit dependency on Issue #1434 uncertainty/coverage reporting.
* [Issue #1236 Optimizer Adversarial Sampler](issue_1236_optimizer_adversarial_sampler.md) -
  Optuna-backed feedback sampler pilot, synthetic comparison helper, and non-paper-facing evidence
  boundary.
* [Issue #1294 Seed-Sensitivity Perturbations](issue_1294_seed_sensitivity_perturbations.md) -
  bounded timing/speed perturbation grids for adversarial seed-sensitivity replays.
* [Issue #923 Multi-Ped Adversarial Candidate Schema](issue_923_multi_ped_adversarial_schema.md) -
  schema-only first slice under #870 for scripted multi-pedestrian adversarial candidates.
* [Issue #936 Multi-Ped Adversarial Overrides](issue_936_multi_ped_adversarial_overrides.md)
  records the pure-data materializer from `adversarial-multi-ped.v1` configs to scenario-loader
`single_pedestrians` override dictionaries, stacked on the issue #923 schema PR.
* [Issue #944 Multi-Ped Adversarial Scenario Payload](issue_944_multi_ped_adversarial_scenario_payload.md)
  adds a template-merging manifest payload materializer for `adversarial-multi-ped.v1` configs, 
  stacked on the issue #936 override materializer.
* [Issue #870 Multi-Ped Adversarial Runtime Slice](issue_870_multi_ped_adversarial_runtime.md)
  adds a fail-closed config-to-`RobotSimulationConfig` runtime path with N>1 reset/step proof and
  records the follow-up materialized policy-analysis smoke while keeping certification and benchmark
  promotion out of scope.
* [Issue #1015 Multi-Ped Adversarial Family Smoke](issue_1015_multi_ped_family_smoke.md)
  adds group-squeeze and doorway-blocker development smoke fixtures with deterministic reset/step
  proof and explicit non-benchmark-frozen episode metadata.
* [Issue 868 Scenario Certification](issue_868_scenario_certification.md) - `scenario_cert.v1`
  scope, public surfaces, validation path, and known limits.
* [Issue #930 CARLA T0 Neutral Export Schema](issue_930_carla_t0_export_schema.md)
  records the import-safe `robot_sf_carla_bridge` package, `carla-replay-export.v1` schema, and
  missing-CARLA `not-available` guard for future oracle replay work.
* [Issue #872 CARLA Oracle Replay Bridge Status](issue_872_carla_oracle_replay_bridge_status.md)
  records why the bounded CARLA replay/parity parent could close on 2026-05-25 while keeping setup,
  adapted, native/aligned, metric-parity, and transfer claims distinct.
* [Issue #1485 CARLA Transfer-Boundary Follow-Up](issue_1485_carla_transfer_boundary_follow_up.md)
  preserves the post-closure transfer-boundary taxonomy and defers any broader multi-scenario CARLA
  replay campaign to a separate benchmark issue.
* [Issue #1444 CARLA Coordinate Alignment Contract (2026-05-22)](issue_1444_carla_coordinate_alignment_contract.md)
  defines the conservative replay-mode taxonomy (`native`, `aligned`, `adapted`, `failed`,
  `not-available`) and projection tolerances required before any Robot-SF/CARLA metric parity claim.
* [Issue #1430 CARLA Live Replay Parity 2026-05-21](issue_1430_carla_live_parity.md)
  records the post-#1329 live CARLA rerun on `imech156-u`, the fail-closed robot-spawn blocker,
  and the conservative unavailable parity report.
* [Issue #934 CARLA T0 Export Builder API](issue_934_carla_t0_export_builder.md)
  adds typed, schema-validated builder objects for `carla-replay-export.v1` payload construction, 
  stacked on the issue #930 bridge package.
* [Issue #940 CARLA T0 Export Read Helper](issue_940_carla_t0_export_read_helper.md)
  adds a schema-validating `read_export_payload(...)` boundary for exported replay JSON, stacked on
  the issue #934 builder API.
* [Issue #942 CARLA T0 MapDefinition Adapter](issue_942_carla_t0_map_definition_adapter.md)
  converts already-certified Robot-SF `MapDefinition` objects into schema-valid neutral export
  payloads, stacked on the issue #940 read helper.
* [Issue #946 CARLA T0 Scenario Entry Export](issue_946_carla_t0_scenario_entry_export.md)
  exports one scenario-loader entry through `scenario_cert.v1` into a neutral CARLA T0 payload, 
  stacked on the issue #942 map-definition adapter.
* [Issue #948 CARLA T0 Scenario File Export](issue_948_carla_t0_scenario_file_export.md)
  batch-loads scenario manifests into ordered neutral export payload records, stacked on the
  Issue #946 scenario-entry helper.
* [Issue #950 CARLA T0 Export Record Writer](issue_950_carla_t0_export_record_writer.md)
  writes ordered neutral export records to deterministic JSON files plus a local manifest, stacked
  on the issue #948 scenario-file helper.
* [Issue #952 CARLA T0 Export CLI](issue_952_carla_t0_export_cli.md)
  adds a CARLA-free command-line wrapper over scenario-file export and deterministic record writing, 
  stacked on the issue #950 record writer.
* [Issue #954 CARLA T0 Export CLI Packaging](issue_954_carla_t0_export_cli_packaging.md)
  exposes the issue #952 CLI as an installable project script while keeping the package CARLA-free.
* [Issue #956 CARLA T0 Export Manifest Reader](issue_956_carla_t0_export_manifest_reader.md)
  adds a local manifest reader for issue #950 export records, stacked on the issue #954 CLI package
  surface.
* [Issue #958 CARLA T0 Manifest Validation CLI](issue_958_carla_t0_manifest_validation_cli.md)
  exposes the issue #956 manifest reader through a CARLA-free project script.
* [Issue #960 CARLA T0 Manifest Payload Paths](issue_960_carla_t0_manifest_payload_paths.md)
  resolves issue #950 manifest entries to local payload paths while rejecting unsafe path escapes.
* [Issue #928 CARLA T0/T1 Oracle Replay Contract](issue_928_carla_t0_t1_replay_contract.md)
  documents the first CARLA transfer boundary: neutral export first, optional oracle replay later, 
  and fail-closed `not-available` / `failed` statuses instead of fallback parity claims.
* [Issue #962 CARLA T0 Manifest Payload Loader](issue_962_carla_t0_manifest_payload_loader.md)
  loads and validates all payloads referenced by a local T0 export manifest, stacked on the
  Issue #960 path resolver.
* [Issue #964 CARLA T0 Batch Validation CLI](issue_964_carla_t0_batch_validation_cli.md)
  exposes the issue #962 manifest payload loader through a CARLA-free batch validation project
  script.
- [Issue #966 CARLA T0 Batch Validation JSON Summary](issue_966_carla_t0_batch_validation_json_summary.md)
  adds deterministic machine-readable output to the CARLA-free batch validation CLI.
- [Issue #968 CARLA Runtime Availability Guard](issue_968_carla_runtime_availability_guard.md)
  adds a strict optional-CARLA import guard for future replay entry points.
- [Issue #970 CARLA Availability Check CLI](issue_970_carla_availability_check_cli.md)
  exposes CARLA bridge availability metadata through a CARLA-free project script.
- [Issue #972 CARLA Availability CLI Require Mode](issue_972_carla_availability_cli_require_mode.md)
  adds fail-closed availability checking for CARLA-dependent setup gates.
- [Issue #974 CARLA Availability Boolean Field](issue_974_carla_availability_boolean_field.md)
  adds an explicit boolean to CARLA availability metadata and CLI JSON output.
- [Issue #976 CARLA Availability Schema Version](issue_976_carla_availability_schema_version.md)
- [Issue #1270 Hazard Traceability Mapping](issue_1270_hazard_traceability.md)
  adds a schema version to CARLA availability metadata for stable script consumption.
- [Issue #1269 ODD Contract Schema](issue_1269_odd_contract_schema.md)
  adds a versioned ODD metadata contract for benchmark and falsification evidence boundaries.
- [Issue #978 CARLA Availability JSON Schema](issue_978_carla_availability_json_schema.md)
  adds a JSON Schema for validating CARLA availability metadata.
- [Issue #980 CARLA Availability Schema CLI](issue_980_carla_availability_schema_cli.md)
  exposes the CARLA availability JSON Schema through `robot-sf-check-carla --schema`.
- [Issue #982 CARLA T0 Export Schema CLI](issue_982_carla_t0_export_schema_cli.md)
  exposes the CARLA T0 neutral export JSON Schema through `robot-sf-export-carla-t0 --schema`.
- [Issue #984 CARLA T0 Export Manifest Schema](issue_984_carla_t0_export_manifest_schema.md)
  adds a packaged JSON Schema for `carla-replay-export-manifest.v1` metadata.
- [Issue #986 CARLA T0 Manifest Schema CLI](issue_986_carla_t0_manifest_schema_cli.md)
  exposes the CARLA T0 export manifest JSON Schema through
  `robot-sf-validate-carla-t0-manifest --schema`.
- [Issue #988 CARLA T0 Batch Summary Version](issue_988_carla_t0_batch_summary_version.md)
  adds a schema version marker to `robot-sf-validate-carla-t0-batch --json` output.
- [Issue #990 CARLA T0 Batch Summary Schema](issue_990_carla_t0_batch_summary_schema.md)
  adds a packaged JSON Schema for `carla-replay-export-batch-validation-summary.v1`.
- [Issue #992 CARLA T0 Batch Summary Schema CLI](issue_992_carla_t0_batch_summary_schema_cli.md)
  exposes the CARLA T0 batch validation summary JSON Schema through
  `robot-sf-validate-carla-t0-batch --schema`.
- [Issue #994 CARLA Bridge Schema Catalog API](issue_994_carla_schema_catalog_api.md)
  adds an import-safe catalog for CARLA bridge schema names, versions, and loader helpers.
- [Issue #996 CARLA Bridge Schema Catalog CLI](issue_996_carla_schema_catalog_cli.md)
  exposes the CARLA bridge schema catalog through `robot-sf-catalog-carla-schemas`.
- [Issue #998 CARLA Bridge Schema Catalog Schema](issue_998_carla_schema_catalog_schema.md)
  adds a packaged JSON Schema for `carla-bridge-schema-catalog.v1` metadata.
- [Issue #1000 CARLA Bridge Schema Catalog Schema CLI](issue_1000_carla_schema_catalog_schema_cli.md)
  exposes the CARLA bridge schema catalog JSON Schema through
  `robot-sf-catalog-carla-schemas --schema`.
- [Issue #1076 AMV Paper-Defense Backlog Tracker](issue_1076_amv_paper_defense_backlog.md)
  records the approved AMV backlog child issues, filing waves, dependency notes, and the current
  Wave 1 PR linkage without treating follow-up waves as submission blockers.
- [Issue #1003 CARLA T1 Oracle Replay Smoke](issue_1003_carla_t1_oracle_smoke.md)
  adds a setup-only T1 oracle replay smoke command for one T0 export manifest payload, with
  schema-catalog validation and fail-closed `not-available` behavior when CARLA is absent.
- [Issue #1179 CARLA Docker Runtime](issue_1179_carla_docker_runtime.md)
  records the pinned `carlasim/carla:0.9.16` Docker runtime interface, preflight/smoke command,
  local no-sudo Docker blocker, and boundary before true live replay semantics.
- [Issue #1169 CARLA Live T1 Oracle Replay](issue_1169_carla_live_replay.md)
  adds the Docker-backed live replay command, records a real CARLA `0.9.16` connection on
  `Town10HD_Opt`, and fails closed on static-geometry replay for the inherited #1111 payload.

## DreamerV3 Notes

* [DreamerV3 Program Full Handoff (2026-04-28)](dreamerv3_program_full_handoff_2026_04_28.md)
  Consolidated execution plan for issues #578, #608, #609, #782, and #789.
* [DreamerV3 BR-08 Full Progress (2026-04-29)](dreamerv3_br08_full_progress_2026_04_29.md)
  Run-level outcome and diagnostics summary for Slurm 12159.
* [DreamerV3 Program Close-Out (2026-04-30)](dreamerv3_program_close_out_2026_04_30.md)
  Program-level stop decision and closure rationale after the probe/gate/full sequence.
* [Issue 782: DreamerV3 world-model pretraining design](issue_782_dreamerv3_pretraining_design.md)
  Inventory of reusable rollout sources plus the recommended proof-first pretraining path.
* [Issue #1190 DreamerV3 checkpoint import boundary probe](issue_1190_dreamerv3_checkpoint_import_boundary.md)
  Fail-closed result for the BR-08 warm-start import question: Ray 2.53.0 exposes full
  Algorithm/RLModule restore, not a clean Robot SF world-model import contract.
* [Issue 789: DreamerV3 multimodal encoder stop note](issue_789_dreamer_multimodal_encoder.md)
  Fail-closed investigation result for mixed observation spaces on Ray 2.53.0 DreamerV3.

## Hybrid-Learning Program Notes

* [Issue #1499 Hard-Guarded Hybrid-Learning Evidence Matrix Schema](issue_1499_hybrid_evidence_matrix_schema.md)
  defines the canonical evidence matrix schema that the synthesis consumer (#1489) will reference
  when component campaigns complete. Includes field contracts, evidence-tier vocabulary,
  non-evidence/failure-mode enumeration, guard-authority constraints, and consumer rules.
