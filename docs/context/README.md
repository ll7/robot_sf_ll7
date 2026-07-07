# Context Notes Workflow

`docs/context/` is the repository's Markdown knowledge base for issue execution history, durable
agent handoff, and reusable reasoning that should not be trapped in chat or PR text.

For broad context lookup, start with [INDEX.md](INDEX.md). It is the retrieval-first catalog for
current entry points, status rules, and curated context-pack scopes. This README remains the note
maintenance workflow and full discoverability surface.

Use this directory for non-trivial insights, decisions, tradeoffs, validation notes, and execution
context that future contributors or agents are likely to need again.

For failure-mechanism work, start with
[issue_2012_failure_mechanism_classifier.md](issue_2012_failure_mechanism_classifier.md) for the
paired-horizon executable classifier and
[issue_2220_failure_mechanism_taxonomy.md](issue_2220_failure_mechanism_taxonomy.md) for the
mechanism-level interpretation vocabulary and confidence classes.

## Archive

`docs/context/archive/` preserves historical notes that are no longer active retrieval entry
points. Do not delete archived notes unless a maintainer explicitly removes that provenance; keep
catalog rows as `historical` or `superseded` with `replacement` pointers when a current successor
exists.

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

* Retrieval-first context index: [INDEX.md](INDEX.md)
* Machine-readable context catalog: [catalog.yaml](catalog.yaml)
* Repository rule: [AGENTS.md](../../AGENTS.md)
* Contributor workflow: [docs/dev_guide.md](../dev_guide.md)
* Docs index entry: [docs/README.md](../README.md)
* AI-facing orientation: [docs/ai/repo_overview.md](../ai/repo_overview.md)
* Researcher's Guide (question → evidence tier → manifest → validation → publish):
  [docs/researchers_guide.md](../researchers_guide.md)
* Research-engine month-one synthesis (#3057 landed-vs-pending, synthesis only):
  [research_month_one_synthesis_2026-06.md](research_month_one_synthesis_2026-06.md)
* Issue #2946 mechanism-evidence figure pack: [issue_2946_mechanism_figure_pack.md](issue_2946_mechanism_figure_pack.md)
* Issue #3279 Social Mini-Game scenario families (v0): [issue_3279_social_mini_game_families.md](issue_3279_social_mini_game_families.md)
* Goal-driven agent loops:
  [goal_driven_agent_loops_2026-05-13.md](goal_driven_agent_loops_2026-05-13.md)
* Skill Consolidation Audit Issue #3189 2026-06-20:
  [skill_consolidation_audit_2026-06-20.md](skill_consolidation_audit_2026-06-20.md)
* Issue #3472 PR body contract workflow:
  [issue_3472_pr_body_contracts.md](issue_3472_pr_body_contracts.md)

* Issue #3385 camera-ready decomposition closure audit:

  [issue_3385_closure_audit_2026_07_04.md](issue_3385_closure_audit_2026_07_04.md)
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
* Open-issues training split audit 2026-05-30:
  [open_issues_training_split_audit_2026-05-30.md](open_issues_training_split_audit_2026-05-30.md)
  (historical snapshot; current queue routing is superseded by live GitHub labels and
  [issue_1776_state_label_routing.md](issue_1776_state_label_routing.md)).
* Open-issue execution improvement plan 2026-05-30:
  [open_issue_execution_improvement_plan_2026-05-30.md](open_issue_execution_improvement_plan_2026-05-30.md)
* Issue #2272 ORCA-residual launch packet status:
  [issue_2272_orca_residual_launch_packet_status.md](issue_2272_orca_residual_launch_packet_status.md)
* Issue #2311 ORCA-residual lane decision:
  [issue_2311_orca_residual_lane_decision.md](issue_2311_orca_residual_lane_decision.md)
* Issue #1358 ORCA-residual parent closure audit:
  [issue_1358_closure_audit_2026-07-07.md](issue_1358_closure_audit_2026-07-07.md)
* Issue #2271 oracle-imitation trace preflight:
  [issue_2271_oracle_imitation_trace_preflight.md](issue_2271_oracle_imitation_trace_preflight.md)
* Issue #2620 oracle-imitation artifact access audit:
  [issue_2620_oracle_artifact_access.md](issue_2620_oracle_artifact_access.md)
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
* Issue #1554 S20/S30 Seed-Budget Bundle:
  [issue_1554_s20_s30_seed_budget.md](issue_1554_s20_s30_seed_budget.md)
* Issue #1508 CARLA native/aligned eligibility audit:
  [issue_1508_carla_native_aligned_eligibility.md](issue_1508_carla_native_aligned_eligibility.md)
* Issue #1509 CARLA native fixture certification:
  [issue_1509_carla_native_fixture_certification.md](issue_1509_carla_native_fixture_certification.md)
* Issue #2276 CARLA parity lane decision:
  [issue_2276_carla_parity_lane_decision.md](issue_2276_carla_parity_lane_decision.md)
* Issue #3028 external simulator bridge preflight:
  [issue_3028_external_simulator_bridge_preflight.md](issue_3028_external_simulator_bridge_preflight.md)
* Issue #4289 ATC public external-data surface terminality:
  [issue_4289_atc_public_surface_terminality.md](issue_4289_atc_public_surface_terminality.md)
* Issue #1894 SLURM job finalizer:
  [issue_1894_slurm_job_finalizer.md](issue_1894_slurm_job_finalizer.md)
* Issue #3075 durable artifact backend decision (W&B):
  [issue_3075_durable_artifact_backend.md](issue_3075_durable_artifact_backend.md)
* Issue #3425 SLURM (Simple Linux Utility for Resource Management)-to-Claim Blocker:
  [issue_3425_slurm_to_claim_blocker.md](issue_3425_slurm_to_claim_blocker.md)
  records why the requested vertical slice cannot be submitted from the current local machine and
  the smallest valid external action.
* Issue #4520 serial benchmark graphics processing unit (GPU) memory closure audit:

  [issue_4520_gpu_memory_closure_audit_2026-07-05.md](issue_4520_gpu_memory_closure_audit_2026-07-05.md)

  records that merged PR #4528 delivered CPU-testable serial-arm teardown; no GPU
  campaign, Slurm submission, or benchmark claim was run by audit.
* Issue #4546 Social Navigation Benchmark (SocNavBench) ETH stage/SVG closure audit:
[issue_4546_closure_audit_2026-07-06.md](issue_4546_closure_audit_2026-07-06.md)
records real external-root source staging, converter input/output hashes, and
parser-smoke validation for `maps/svg_maps/socnavbench/socnavbench_eth.svg`.
* Issue #2232 planner mechanism transfer benchmark protocol:
[issue_2232_planner_mechanism_transfer_benchmark.md](issue_2232_planner_mechanism_transfer_benchmark.md)
* Issue #3064 behavior-variant inventory:
  [issue_3064_behavior_variants_inventory.md](issue_3064_behavior_variants_inventory.md)
  records the fail-closed classification of native Social Force, AMMV-aware Social Force, and
  Social-Navigation-PyEnvs adapter-backed behavior variants for benchmark selection.
* Issue #3063 campaign comparison report:
  [issue_3063_campaign_comparison_report.md](issue_3063_campaign_comparison_report.md)
  records the analysis-only result-store report path, fixture proof, row-status caveats, and
  follow-up boundary before any benchmark or paper-facing claim.
* Issue #3062 campaign manifest flow:
  [issue_3062_campaign_manifest_flow.md](issue_3062_campaign_manifest_flow.md)
  records the reusable research-campaign manifest contract, example manifest, output boundary, and
  validation path before runner-specific automation treats manifests as benchmark evidence.
* Issue #3142 fast-pysf force optimization:
  [issue_3142_fast_pysf_force_optimization.md](issue_3142_fast_pysf_force_optimization.md)
  records the bounded semantics-preserving force-computation cleanup, diagnostic performance
  boundary, and tracked compact summary.
* Issue #3146/#3164 forecast replay fixture suite:
  [issue_3146_forecast_replay_fixture_suite.md](issue_3146_forecast_replay_fixture_suite.md)
  records the scenario-diverse frozen-policy diagnostic forecast replay suite, full variant
  matrix row classifications, tracked #3164 evidence summary, and the negative result that
  non-`none` variants collapse under shared replay braking.
* Issue #3200 pedestrian-density runtime smoke:
  [issue_3200_density_runtime_smoke_summary.json](evidence/issue_3200_density_runtime_smoke_summary.json)
  records the diagnostic-only same-seed smoke over top coverage-novel and lowest-novelty comparator
  rows, with all rows classified `horizon_exhausted` and no benchmark claim promoted.
* Issue #3281 naturalistic VRU priors:
  [issue_3281_naturalistic_vru_priors.md](issue_3281_naturalistic_vru_priors.md)
  records the additive generated-manifest prior metadata, manifest-quality naturalness filters, and
  plausible-hard-case vs stress-only claim boundary.
* Issue #4360 adversarial dispatchable inventory:
  [issue_4360_adversarial_dispatchable_inventory.md](issue_4360_adversarial_dispatchable_inventory.md)
  records current adversarial pedestrian hooks, repeatable seeds/configs, runner assumptions,
  and runbook boundary for the dispatchable half only.
* Issue #3014 Evidence Catalog Backlog 2026-06-19:
  [issue_3014_evidence_catalog_backlog.md](issue_3014_evidence_catalog_backlog.md)
  records the current uncovered evidence-bundle count and split strategy for catalog cleanup.
* Issue #4404 trace-capable horizon-600 (`h600`) runtime-scaling context:
  [issue_4404_trace_capable_h600_runtime_scaling.md](issue_4404_trace_capable_h600_runtime_scaling.md)
  records static config-derived workload surface for the issue #4206 trace-capable h600 re-run and
  the explicit no-submit/no-claim boundary.
* Active research-lane scientific states:
  [research_lane_states.md](research_lane_states.md)
* Issue #2571 Active Research Queue:
  [issue_2571_active_research_queue.md](issue_2571_active_research_queue.md)
* Issue #2542 Dissertation Figure/Table Export Bundle:
  [issue_2542_dissertation_export_bundle.md](issue_2542_dissertation_export_bundle.md)
  ([spec](evidence/issue_2542_dissertation_export_bundle/artifact_spec.json),
  [manifest](evidence/issue_2542_dissertation_export_bundle/artifact_manifest.json),
  [checksums](evidence/issue_2542_dissertation_export_bundle/checksums.sha256),
  [campaign table](evidence/issue_2542_dissertation_export_bundle/payload/artifacts/tab_issue_1023_campaign_table.md),
  [scenario-family table](evidence/issue_2542_dissertation_export_bundle/payload/artifacts/tab_issue_1023_scenario_family_breakdown.md)).
  The #3203 refresh is payload-complete but diagnostic only because the row-complete July 1
  campaign still fails the Social Navigation Quality Index (SNQI) rank-alignment contract.
* Issue #2686 Release 0.0.2 Table Evidence Bundle:
  [issue_2686_release_0_0_2_table_bundle.md](issue_2686_release_0_0_2_table_bundle.md)
  ([spec](evidence/issue_2686_release_0_0_2_table_bundle/artifact_spec.json),
  [manifest](evidence/issue_2686_release_0_0_2_table_bundle/artifact_manifest.json),
  [checksums](evidence/issue_2686_release_0_0_2_table_bundle/checksums.sha256))
* Issue #2689 Release Evidence Handoff Snapshot:
  [issue_2689_release_evidence_handoff_2026_06_15.md](issue_2689_release_evidence_handoff_2026_06_15.md)
* Issue #3205 Release Evidence Snapshot Contract:
  [issue_3205_release_evidence_snapshot_contract.md](issue_3205_release_evidence_snapshot_contract.md)
  records the dry-runable release evidence manifest gate, fail-closed missing-input behavior,
  DOI-ready metadata fields, and current diagnostic-only artifact-catalog proof.
* Issue #3294 Release Claim Matrix:
  [release_claim_matrix.md](evidence/issue_3294_release_claim_matrix/release_claim_matrix.md)
  records the v0.1 row-level synthesis over existing release artifacts, leaderboard row-claim
  sidecars, and ODD coverage metadata. It does not add a new benchmark campaign or promote
  fallback, degraded, unavailable, or diagnostic rows as successful benchmark evidence.
* Issue #2536 simulator-speed candidate discovery:
  [issue_2536_speed_discovery.md](issue_2536_speed_discovery.md)
* Issue #2531 AMV trace-boundary decision:
  [issue_2531_amv_trace_boundary.md](issue_2531_amv_trace_boundary.md)
* Dissertation claim-export candidate report:
  [dissertation_claim_export_candidate_report.md](dissertation_claim_export_candidate_report.md)
* Dissertation research bridge:
  [dissertation_research_bridge.md](dissertation_research_bridge.md)
* Dissertation evidence ledger (Issue #2760):
  [dissertation_evidence_ledger.md](dissertation_evidence_ledger.md)
  ([JSON ledger](evidence/issue_2760_dissertation_evidence_ledger/ledger.json),
  [validation test](../../tests/docs/test_dissertation_evidence_ledger.py))
* Dissertation gap report (Issue #2784):
  [dissertation_gap_report.md](dissertation_gap_report.md)
  ([JSON report](evidence/issue_2784_dissertation_gap_report/gap_report.json),
  [Markdown report](evidence/issue_2784_dissertation_gap_report/gap_report.md),
  [validation test](../../tests/docs/test_dissertation_gap_report.py))
* Issue #3426 result-card generator:
  [issue_3426_result_cards.md](issue_3426_result_cards.md)
  records the fail-closed bridge from accepted evidence summaries to dissertation-ready result-card
  Markdown/JSON/LaTeX snippets.
* Negative result register (Issue #2762):
  [negative_result_register.md](negative_result_register.md)
  ([JSON register](evidence/issue_2762_negative_result_register/register.json),
  [validation test](../../tests/docs/test_negative_result_register.py))
* Issue #2221 static-recentering transfer smoke:
  [issue_2221_static_recenter_transfer.md](issue_2221_static_recenter_transfer.md)
* Issue #2282 topology selection instrumentation smoke:
  [issue_2282_topology_selection_instrumentation.md](issue_2282_topology_selection_instrumentation.md)
* Issue #2307 topology score diagnostic:
  [issue_2307_topology_score_diagnostic.md](issue_2307_topology_score_diagnostic.md)
* Issue #2393 topology selection preflight:
  [issue_2393_topology_selection_preflight.md](issue_2393_topology_selection_preflight.md)
* Issue #2403 topology selection-score decision:
  [issue_2403_topology_selection_score_decision.md](issue_2403_topology_selection_score_decision.md)
* Issue #2518 topology near-parity gate diagnostic:
  [issue_2518_topology_near_parity_gate.md](issue_2518_topology_near_parity_gate.md)
* Issue #2530 topology near-parity corrective-behavior smoke:
  [issue_2530_topology_near_parity_corrective_smoke.md](issue_2530_topology_near_parity_corrective_smoke.md)
* Issue #2563 Topology Corrective Revision Proposal 2026-06-07:
  [issue_2563_topology_corrective_revision.md](issue_2563_topology_corrective_revision.md)
* Issue #2570 Topology Revise Status Propagation:
  [issue_2570_topology_revise_status_propagation.md](issue_2570_topology_revise_status_propagation.md)
* Issue #2600 Topology Revision Decision 2026-06-11:
  [issue_2600_topology_revision_decision.md](issue_2600_topology_revision_decision.md)
* Issue #2540 Topology Primary-Route Reuse-Penalty Diagnostic:
  [issue_2540_topology_reuse_penalty_diagnostic.md](issue_2540_topology_reuse_penalty_diagnostic.md)
* Issue #2621 Topology Revision Hypothesis 2026-06-11:
  [issue_2621_topology_revision_hypothesis.md](issue_2621_topology_revision_hypothesis.md)
* Issue #2624 Topology Reuse-Penalty Paired Diagnostic Gate 2026-06-11:
  [issue_2624_topology_reuse_penalty_gate.md](issue_2624_topology_reuse_penalty_gate.md)
  ([summary](evidence/issue_2624_topology_reuse_penalty_gate/summary.json))
* Issue #2660 Topology Successor Gate After Reuse-Penalty Regression (superseded by Issue #2704):
  [issue_2660_topology_successor_gate.md](archive/issue_2660_topology_successor_gate.md)
* Issue #2704 Progress-Gated Topology Successor Diagnostic:
  [issue_2704_progress_gated_topology_successor.md](issue_2704_progress_gated_topology_successor.md)
  ([summary](evidence/issue_2704_progress_gated_topology_successor/summary.json))
* Issue #2716 topology reselection cross-slice diagnostic:
  [issue_2716_topology_reselection_cross_slice.md](issue_2716_topology_reselection_cross_slice.md)
  ([summary](evidence/issue_2716_topology_reselection_cross_slice/summary.json))
* Issue #2742 topology reselection successor launch packet:
  [issue_2742_topology_reselection_successor.md](issue_2742_topology_reselection_successor.md)
  ([summary](evidence/issue_2742_topology_reselection_successor/summary.json))
* Issue #2752 topology reselection mechanism diagnosis:
  [issue_2752_topology_reselection_mechanism.md](issue_2752_topology_reselection_mechanism.md)
  ([evidence](evidence/issue_2752_topology_reselection_mechanism/README.md))
* Issue #3463 topology corrective behaviors integration report:
  [issue_3463_topology_corrective_behaviors.md](issue_3463_topology_corrective_behaviors.md)
* Issue #2801 topology successor recommendation:
  [issue_2801_topology_successor_recommendation.md](issue_2801_topology_successor_recommendation.md)
  ([summary](evidence/issue_2801_topology_successor_recommendation/summary.json))
* Issue #2804 Non-Topology Successor Launch Packet:
  [issue_2804_non_topology_successor.md](issue_2804_non_topology_successor.md)
  ([summary](evidence/issue_2804_non_topology_successor/summary.json))
* Issue #2706 topology lane synthesis after progress-gated successor:
  [issue_2706_topology_lane_synthesis.md](issue_2706_topology_lane_synthesis.md)
* Mechanism closure status:
  [mechanism_closure_status.md](mechanism_closure_status.md)
* Issue #2389 mechanism-aware evaluation thread:
  [issue_2389_mechanism_aware_evaluation_thread.md](issue_2389_mechanism_aware_evaluation_thread.md)
* Issue #2266 Static-Recenter Activation Diagnostic 2026-06-05:
  [issue_2266_static_recenter_activation.md](issue_2266_static_recenter_activation.md)
* Issue #2306 Static-Recenter Activation Trace 2026-06-05:
[issue_2306_static_recenter_activation_trace.md](issue_2306_static_recenter_activation_trace.md)
* Issue #2402 static-recenter activation decision:
  [issue_2402_static_recenter_activation_decision.md](issue_2402_static_recenter_activation_decision.md)
* Issue #2438 static-recenter activation closure:
  [issue_2438_static_recenter_activation_closure.md](issue_2438_static_recenter_activation_closure.md)
* Issue #2566 static-recenter inactive propagation:
  [issue_2566_static_recenter_inactive_propagation.md](issue_2566_static_recenter_inactive_propagation.md)
* Issue #2261 Static-Recenter Slice-Local Explanation:
[issue_2261_static_recenter_slice_local.md](issue_2261_static_recenter_slice_local.md)
* Issue #1573 Root-Layout Inventory:
  [issue_1573_root_layout_inventory.md](issue_1573_root_layout_inventory.md)
* Root layout structured migration 2026-06-01:
  [root_layout_structured_migration_2026-06-01.md](root_layout_structured_migration_2026-06-01.md)
* Issue #2035 path-reference audit:
  [issue_2035_path_reference_audit.md](issue_2035_path_reference_audit.md)
* Issue #1584 SocNavBench Unavailable Row Policy (2026-05-28):
  [issue_1584_socnav_unavailable_row_policy.md](issue_1584_socnav_unavailable_row_policy.md)
* Issue #2397 SocNavBench Control-Pipeline Asset Status (2026-06-06):
  [issue_2397_socnavbench_control_status_2026-06-06.md](issue_2397_socnavbench_control_status_2026-06-06.md)
* Issue #1583 High-Risk Root Path Boundaries (superseded by
  [root_layout_structured_migration_2026-06-01.md](root_layout_structured_migration_2026-06-01.md);
  provenance only):
  [issue_1583_high_risk_root_boundaries.md](archive/issue_1583_high_risk_root_boundaries.md)
* Issue #1636 benchmark metric semantics:
  [issue_1636_benchmark_metric_semantics.md](issue_1636_benchmark_metric_semantics.md)
* Issue #1634 SocNav Module Split Plan (2026-05-28):
  [issue_1634_socnav_split_plan.md](issue_1634_socnav_split_plan.md)
* Issues #1598/#1599 Root Compatibility Decisions (2026-05-28) (superseded by
  [root_layout_structured_migration_2026-06-01.md](root_layout_structured_migration_2026-06-01.md);
  provenance only):
  [issue_1598_1599_root_compatibility_decisions.md](archive/issue_1598_1599_root_compatibility_decisions.md)
* Issue #1504 ego-conditioned feature contract:
  [issue_1504_ego_feature_contract.md](issue_1504_ego_feature_contract.md)
* Issue #1543 predictive v2 negative audit:
  [issue_1543_predictive_v2_negative_audit.md](issue_1543_predictive_v2_negative_audit.md)
* Issue #1490 predictive-v2 closure audit:
  [issue_1490_closure_audit.md](issue_1490_closure_audit.md)
* Issue #2275 predictive-v2 fate decision:
  [issue_2275_predictive_v2_fate.md](issue_2275_predictive_v2_fate.md)
* Issue #2411 predictive-v2 child classification:
  [issue_2411_predictive_v2_child_classification.md](issue_2411_predictive_v2_child_classification.md)
* Issue #3254 predictive crossing-conflict negative result:
  [issue_3254_predictive_crossing_conflict_negative_result.md](issue_3254_predictive_crossing_conflict_negative_result.md)
* Issue #2468 Adversarial Scenario Generation Roadmap:
  [issue_2468_adversarial_generation_roadmap.md](issue_2468_adversarial_generation_roadmap.md)
* Issue #2470 RL Adversarial Pedestrian Policy Scope:
  [issue_2470_rl_adversarial_pedestrian_policy.md](issue_2470_rl_adversarial_pedestrian_policy.md)
* Issue #2471 Diffusion Scenario Generation Feasibility Scope:
  [issue_2471_diffusion_scenario_generation_feasibility.md](issue_2471_diffusion_scenario_generation_feasibility.md)
* Issue #2472 Intent-Conditioned Pedestrian Behavior Scope:
  [issue_2472_intent_conditioned_behavior.md](issue_2472_intent_conditioned_behavior.md)
* Issue #2524 Adversarial Scenario Manifest Generation:
  [issue_2524_adversarial_manifests.md](issue_2524_adversarial_manifests.md)
* Issue #2562 Adversarial Manifest Planner Smoke:
  [issue_2562_adversarial_manifest_smoke.md](issue_2562_adversarial_manifest_smoke.md)
* Issue #2618 Generated Adversarial Manifest Planner Smoke:
  [issue_2618_adversarial_manifest_smoke.md](issue_2618_adversarial_manifest_smoke.md)
* Issue #2725 Generator-Readiness Contract:
  [issue_2725_generator_readiness.md](issue_2725_generator_readiness.md)
* Issue #2565 Uncertainty-Aware Stream-Gap Gating Smoke:
  [issue_2565_uncertainty_gating_smoke.md](issue_2565_uncertainty_gating_smoke.md)
* Issue #2538 ScenarioBelief Planner Projection Smoke:
  [issue_2538_scenario_belief_planner_projection.md](issue_2538_scenario_belief_planner_projection.md)
* Issue #2606 Uncertainty Gate Evaluation:
  [issue_2606_uncertainty_gate_evaluation.md](issue_2606_uncertainty_gate_evaluation.md)
* Issue #2526 Cyclist-Like VRU Smoke:
  [issue_2526_cyclist_vru_smoke.md](issue_2526_cyclist_vru_smoke.md)
* Issue #2527 Waiting-Then-Crossing Fixture:
  [issue_2527_waiting_crossing_fixture.md](issue_2527_waiting_crossing_fixture.md)
* Issue #2564 Signal-State Proxy Smoke:
  [issue_2564_signal_state_proxy_smoke.md](issue_2564_signal_state_proxy_smoke.md)
* Issue #2662 Signal-State Promotion Contract:
  [issue_2662_signal_state_promotion_contract.md](issue_2662_signal_state_promotion_contract.md)
* Issue #2473 Cyclist Interaction Benchmark Scope:
  [issue_2473_cyclist_interaction_benchmark.md](issue_2473_cyclist_interaction_benchmark.md)
* Issue #2459 SocNavBench / HuNavSim Mapping:
  [issue_2459_socnavbench_hunavsim_mapping.md](issue_2459_socnavbench_hunavsim_mapping.md)
* Issue #2928 SocNavBench / HuNavSim Metric Correspondence:
  [issue_2928_socnavbench_hunavsim_metric_correspondence.md](issue_2928_socnavbench_hunavsim_metric_correspondence.md)
* Issue #2930 External Benchmark Positioning:
  [issue_2930_external_benchmark_positioning.md](issue_2930_external_benchmark_positioning.md)
* Issue #3290 Simulation Model Credibility Checklist:
  [issue_3290_simulation_model_credibility_checklist.md](issue_3290_simulation_model_credibility_checklist.md)
* Issue #2442 Navground planner-zoo assessment:
  [issue_2442_navground_assessment.md](issue_2442_navground_assessment.md)
* Issue #2550 (2026-06-11): Navground Adapter Spike Report:
  [issue_2550_navground_adapter_spike.md](issue_2550_navground_adapter_spike.md)
* Issue #2476 Multimodal Prediction Benchmark Contract:
  [issue_2476_multimodal_prediction_benchmark.md](issue_2476_multimodal_prediction_benchmark.md)
* Issue #2496 Multimodal Prediction Contract Smoke:
  [issue_2496_multimodal_prediction_smoke.md](issue_2496_multimodal_prediction_smoke.md)
* Issue #1542 manuscript claim evidence map:
  [issue_1542_manuscript_claim_evidence_map.md](issue_1542_manuscript_claim_evidence_map.md)

* Issue #2228 research dashboard:

  [issue_2228_research_dashboard.md](issue_2228_research_dashboard.md)

* Issue #2153 research-v1 evidence map and claim gate:

  [issue_2153_research_v1_evidence_map.md](issue_2153_research_v1_evidence_map.md)

* Issue #2269 research-v1 trace case selection:

  [issue_2269_research_v1_trace_case_selection.md](issue_2269_research_v1_trace_case_selection.md)

* Issue #2280 research-v1 first trace review:

  [issue_2280_research_v1_first_trace_review.md](issue_2280_research_v1_first_trace_review.md)

* Issue #2281 research-v1 trace review pack:

  [issue_2281_research_v1_trace_review_pack.md](issue_2281_research_v1_trace_review_pack.md)

* Issue #2309 AMV trace export blocker (historical; partially superseded by Issue #2405):

  [issue_2309_amv_trace_export_blocker.md](issue_2309_amv_trace_export_blocker.md)

* Issue #2405 AMV step-export decision:

  [issue_2405_amv_step_export_decision.md](issue_2405_amv_step_export_decision.md)

* Issue #2428 AMMV mechanism trace panels:

  [issue_2428_mechanism_trace_panels.md](issue_2428_mechanism_trace_panels.md)

* Issue #2923 Mechanism Trace v1 Schema:

  [issue_2923_mechanism_trace_schema.md](issue_2923_mechanism_trace_schema.md)

* Issue #2976 Mechanism Trace ORCA Residual Producer:

  [issue_2976_mechanism_trace_orca_residuals.md](issue_2976_mechanism_trace_orca_residuals.md)

* Issue #2430 AMMV trace annotation decision:

  [issue_2430_ammv_trace_annotation.md](issue_2430_ammv_trace_annotation.md)

* Issue #2432 AMMV trace selection decision:

  [issue_2432_ammv_trace_selection.md](issue_2432_ammv_trace_selection.md)

* Issue #2434 AMMV scenario sweep decision:

  [issue_2434_ammv_scenario_sweep.md](issue_2434_ammv_scenario_sweep.md)

* Issue #2463 Mechanism Signal Checker:

  [issue_2463_mechanism_signal_checker.md](issue_2463_mechanism_signal_checker.md)

* Issue #2543 trace failure predicates:

  [issue_2543_trace_failure_predicates.md](issue_2543_trace_failure_predicates.md)

* Issue #2667 trace failure predicate tables:

  [issue_2667_trace_failure_predicate_tables.md](issue_2667_trace_failure_predicate_tables.md)

* Issue #2688 trace predicate benchmark matrix:

  [issue_2688_trace_predicate_matrix.md](issue_2688_trace_predicate_matrix.md)

* Issue #3278 real micromobility trace validation contract:

  [issue_3278_real_trace_validation_contract.md](issue_3278_real_trace_validation_contract.md)

* Issue #2544 mechanism-aware suite smoke:

  [issue_2544_mechanism_aware_suite_smoke.md](issue_2544_mechanism_aware_suite_smoke.md)

* Issue #2586 static-deadlock trace fields:

  [issue_2586_static_deadlock_trace_fields.md](issue_2586_static_deadlock_trace_fields.md)

* Issue #2588 static-deadlock controlled trace:

  [issue_2588_static_deadlock_controlled_trace.md](issue_2588_static_deadlock_controlled_trace.md)

* Issue #2590 escape-recenter static-deadlock controlled trace:

  [issue_2590_escape_recenter_static_deadlock_controlled_trace.md](issue_2590_escape_recenter_static_deadlock_controlled_trace.md)

* Issue #2592 static-deadlock active-row h500 horizon sensitivity:

  [issue_2592_static_deadlock_active_row_h500.md](issue_2592_static_deadlock_active_row_h500.md)

* Issue #2594 broader h500 static-deadlock recenter slice:

  [issue_2594_static_deadlock_broader_h500.md](issue_2594_static_deadlock_broader_h500.md)

* Issue #2596 Static-Deadlock Recenter Claim Boundary 2026-06-11:

  [issue_2596_static_deadlock_recenter_claim_boundary.md](issue_2596_static_deadlock_recenter_claim_boundary.md)

* Issue #2547 counterfactual mechanism taxonomy:

  [issue_2547_counterfactual_mechanism_taxonomy.md](issue_2547_counterfactual_mechanism_taxonomy.md)

* Issue #2659 Manifest Lineage Schema Unification (2026-06-12):

  [issue_2659_lineage_schema_unification.md](issue_2659_lineage_schema_unification.md)

* Issue #2263 mechanism activation report fields:

  [issue_2263_mechanism_activation_report_fields.md](issue_2263_mechanism_activation_report_fields.md)

* Issue #2154 AMMV-aware Social Force model slice:

  [issue_2154_ammv_social_force_model.md](issue_2154_ammv_social_force_model.md)

* Issue #2168 AMMV-aware Social Force paired diagnostic:

  [issue_2168_ammv_social_force_pair_diagnostic.md](issue_2168_ammv_social_force_pair_diagnostic.md)

* Issue #2155 research-v1 AMMV matrix contract:

  [issue_2155_research_v1_ammv_matrix.md](issue_2155_research_v1_ammv_matrix.md)

* Issue #2170 one-factor hybrid component ablation manifest:

  [issue_2170_one_factor_hybrid_component_manifest.md](issue_2170_one_factor_hybrid_component_manifest.md)

* Issue #2172 benchmark worker scaling diagnostic:

  [issue_2172_benchmark_worker_scaling.md](issue_2172_benchmark_worker_scaling.md)

* Issue #2302 benchmark worker scaling continuation:

  [issue_2302_benchmark_worker_scaling.md](issue_2302_benchmark_worker_scaling.md)

* Issue #2304 stress-slice benchmark worker scaling diagnostic:

  [issue_2304_benchmark_worker_scaling_stress.md](issue_2304_benchmark_worker_scaling_stress.md)

* Issue #2214 hot-path optimization synthesis:

  [issue_2214_hot_path_synthesis.md](issue_2214_hot_path_synthesis.md)

* Issue #2174 one-factor hybrid component ablation pilot:

  [issue_2174_one_factor_ablation_pilot.md](issue_2174_one_factor_ablation_pilot.md)

* Issue #2176 remaining one-factor hybrid component h80 comparisons:

  [issue_2176_remaining_one_factor_h80.md](issue_2176_remaining_one_factor_h80.md)

* Issue #2178 selector ORCA-extra h80 rerun:

  [issue_2178_selector_orca_extra_h80.md](issue_2178_selector_orca_extra_h80.md)

* Issue #2180 one-factor hybrid component h500 run:

  [issue_2180_one_factor_h500.md](issue_2180_one_factor_h500.md)

* Issue #2182 component effect synthesis:

  [issue_2182_component_effect_synthesis.md](issue_2182_component_effect_synthesis.md)

* Issue #2225 Learned-Policy Failure Synthesis (2026-06-04):

  [issue_2225_learned_policy_failure_synthesis.md](issue_2225_learned_policy_failure_synthesis.md)

* Issue #2274 hybrid-learning component evidence status matrix:

  [issue_2274_hybrid_component_matrix.md](issue_2274_hybrid_component_matrix.md)

* Issue #2410 hybrid-learning component readiness refresh:

  [issue_2410_hybrid_component_readiness_refresh.md](issue_2410_hybrid_component_readiness_refresh.md)

* Issue #2408 ORCA-residual low-progress analysis:

  [issue_2408_orca_residual_low_progress_analysis.md](issue_2408_orca_residual_low_progress_analysis.md)

* Issue #2231 mechanism-aware ranking comparison:

  [issue_2231_mechanism_aware_ranking.md](issue_2231_mechanism_aware_ranking.md)

* Issue #2224 synthetic AMV actuation ranking diagnostic:

  [issue_2224_amv_actuation_ranking.md](issue_2224_amv_actuation_ranking.md)
* Issue #2268 AMV Timeout Decomposition 2026-06-05:

  [issue_2268_amv_timeout_decomposition.md](issue_2268_amv_timeout_decomposition.md)
* Issue #2308 AMV Timeout Trace Analysis 2026-06-05:

  [issue_2308_amv_timeout_trace_analysis.md](issue_2308_amv_timeout_trace_analysis.md)
* Issue #2404 AMV timeout decomposition decision:

  [issue_2404_amv_timeout_decomposition_decision.md](issue_2404_amv_timeout_decomposition_decision.md)
* Issue #2440 AMV timeout closure:

  [issue_2440_amv_timeout_closure.md](issue_2440_amv_timeout_closure.md)
* Issue #2522 why-first diagnostics:

  [issue_2522_why_first_diagnostics.md](issue_2522_why_first_diagnostics.md)
  ([topology report](evidence/issue_2522_why_first_diagnostics/topology_near_parity_why_first_report.md),
  [AMV report](evidence/issue_2522_why_first_diagnostics/amv_actuation_why_first_report.md))
* Issue #2602 Why-First Report Usefulness:

  [issue_2602_why_first_report_usefulness.md](issue_2602_why_first_report_usefulness.md)
  ([summary](evidence/issue_2602_why_first_report_usefulness/summary.json))
* Issue #2259 AMV Clipping Versus Success Boundary:

  [issue_2259_amv_clipping_success_boundary.md](issue_2259_amv_clipping_success_boundary.md)

* Issue #2234 predictive perturbation criticality protocol:

  [issue_2234_predictive_perturbation_criticality.md](issue_2234_predictive_perturbation_criticality.md)

* Issue #2226 seed sufficiency recommendation:

  [issue_2226_seed_sufficiency_recommendation.md](issue_2226_seed_sufficiency_recommendation.md)

* Issue #1530 optional planner preflight audit:
  [issue_1530_optional_preflight_audit.md](issue_1530_optional_preflight_audit.md)
* Issue #1348 capability-aware map catalog design:
  [issue_1348_capability_map_catalog_design.md](issue_1348_capability_map_catalog_design.md)
* Issue #2001 AMV Actuation Proxy Source Analysis (2026-06-01):
  [issue_2001_amv_actuation_proxy_source_analysis.md](issue_2001_amv_actuation_proxy_source_analysis.md)
* Issue #2230 AMV Actuation Evidence Ladder (2026-06-04):
  [issue_2230_amv_actuation_evidence_ladder.md](issue_2230_amv_actuation_evidence_ladder.md)
* Issue #2011 AMV Actuation Sensitivity Sweep (2026-06-01):
  [issue_2011_amv_actuation_sensitivity_sweep.md](issue_2011_amv_actuation_sensitivity_sweep.md)
* Issue #2104 Component Ablation Pilot (2026-06-02):
  [issue_2104_component_ablation_pilot.md](issue_2104_component_ablation_pilot.md)
* Issue #2014 Simulator Backend Decision Matrix (2026-06-01):
  [issue_2014_simulator_backend_matrix.md](issue_2014_simulator_backend_matrix.md)
* Issue #2016 Webots/Gazebo AMV Prototype Parity Audit (2026-06-01):
  [issue_2016_webots_gazebo_amv_parity_audit.md](issue_2016_webots_gazebo_amv_parity_audit.md)
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
* Issue #1612 parallel observation-space benchmark tracks:
  [issue_1612_observation_track_architecture.md](issue_1612_observation_track_architecture.md)
* Issue #1721 Legacy Benchmark-Track Metadata Audit:
  [issue_1721_benchmark_track_metadata_audit.md](issue_1721_benchmark_track_metadata_audit.md)
* Issue #3469 legacy PPO snapshot parity:
  [issue_3469_legacy_ppo_snapshot_parity.md](issue_3469_legacy_ppo_snapshot_parity.md)
* Issue #1846 metadata worker bridge:
  [issue_1846_metadata_worker_bridge.md](issue_1846_metadata_worker_bridge.md)
* Issue #1659 LiDAR occupancy adapter:
  [issue_1659_lidar_occupancy_adapter.md](issue_1659_lidar_occupancy_adapter.md)
* Issue #1653 CI runtime slice:
  [issue_1653_ci_runtime_slice.md](issue_1653_ci_runtime_slice.md)
* Issue #1690 Root Layout Inventory (superseded by
  [root_layout_structured_migration_2026-06-01.md](root_layout_structured_migration_2026-06-01.md);
  provenance only):
  [issue_1690_root_layout_inventory.md](archive/issue_1690_root_layout_inventory.md)
* Issue #1613 LiDAR Observation Track Setup (2026-05-29):
  [issue_1613_lidar_observation_track.md](issue_1613_lidar_observation_track.md)
* Issue #1614 LiDAR Planner Compatibility Audit (2026-05-29):
  [issue_1614_lidar_planner_compatibility.md](issue_1614_lidar_planner_compatibility.md)
* Issue #1685 dummy learned-policy adapter fixture (2026-05-30):
  [issue_1685_dummy_learned_policy_adapter.md](issue_1685_dummy_learned_policy_adapter.md)
* Issue #1622 Decision Transformer Feasibility (2026-05-30):
  [issue_1622_decision_transformer_feasibility.md](issue_1622_decision_transformer_feasibility.md)
* Issue #1625 Learned Planner Arbitration Assessment (2026-05-30):
  [issue_1625_learned_planner_arbitration.md](issue_1625_learned_planner_arbitration.md)
* Issue #1624 Hybrid-Learning Navigation Architecture (2026-05-30):
  [issue_1624_hybrid_learning_architecture.md](issue_1624_hybrid_learning_architecture.md)
* Issue #1674 Topology-Hypothesis Diagnostics (2026-05-30):
  [issue_1674_topology_hypothesis_diagnostics.md](issue_1674_topology_hypothesis_diagnostics.md)
* Issue #1675 learned risk-surface interface:
  [issue_1675_learned_risk_surface_interface.md](issue_1675_learned_risk_surface_interface.md)
* Issue #1628 Actuation-Aware Learned Navigation for AMVs (2026-05-30):
  [issue_1628_actuation_aware_learned_navigation.md](issue_1628_actuation_aware_learned_navigation.md)
* Issue #1740 Actuation-Aware Learned-Policy Smoke Candidate (2026-05-30):
  [issue_1740_actuation_aware_smoke_candidate.md](issue_1740_actuation_aware_smoke_candidate.md)
* Issue #1720 learned-policy roadmap and issue routing:
  [issue_1720_learned_policy_roadmap.md](issue_1720_learned_policy_roadmap.md)
* Issue #2768 Learned Prediction Readiness Contract:
  [issue_2768_learned_prediction_readiness.md](issue_2768_learned_prediction_readiness.md)
* Issue #2864 Forecast-lane synthesis:
  [issue_2864_forecast_lane_synthesis.md](issue_2864_forecast_lane_synthesis.md)
* Issue #2929 Post-#2883-to-#2893 Forecast and Workflow Synthesis:
  [issue_2929_forecast_workflow_synthesis.md](issue_2929_forecast_workflow_synthesis.md)
* [Issue #3193](https://github.com/ll7/robot_sf_ll7/issues/3193) closed-loop forecast
  falsification paper plan:
  [paper_closed_loop_forecast_falsification.md](docs/context/paper_closed_loop_forecast_falsification.md)
  freezes the #3193 claim, hypotheses, endpoints, thresholds, design matrix,
  child-issue sequence, and figure/table query map before any heavy campaign runs.
* Issue #2902 Live same-seed forecast replay gate:
  [issue_2902_live_forecast_replay_gate.md](issue_2902_live_forecast_replay_gate.md)
* Issue #2944 Native CV-only closed-loop replay smoke:
  [issue_2944_native_cv_closed_loop_smoke.md](issue_2944_native_cv_closed_loop_smoke.md)
* Issue #2941 Native forecast-variant replay:
  [issue_2941_native_forecast_replay.md](issue_2941_native_forecast_replay.md)
* Issue #2960 Forecast planner consumer smoke:
  [issue_2960_forecast_planner_consumer.md](issue_2960_forecast_planner_consumer.md)
* Issue #2966 Planner-consumed forecast slice:
  [issue_2966_planner_consumed_forecast_slice.md](issue_2966_planner_consumed_forecast_slice.md)
* Issue #2937 Horizon/timestep denominator-health fixture repair:
  [issue_2937_horizon_denominator_health.md](issue_2937_horizon_denominator_health.md)
* Issue #2943 Fast-Results Milestone / Claim Map v0 (2026-06-16):
  [issue_2943_fast_results_claim_map_v0.md](issue_2943_fast_results_claim_map_v0.md)
* Issue #2911 ODD Hazard Coverage Matrix v1 evidence (2026-06-17):
  [coverage_matrix.md](evidence/issue_2911_odd_hazard_coverage_2026-06-17/coverage_matrix.md)
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
* Issue #2953 Qwen-RobotNav Metric And Benchmark Audit (2026-06-19):
  [issue_2953_qwen_robotnav_metric_audit.md](issue_2953_qwen_robotnav_metric_audit.md)
* Issue #2468 Adversarial Scenario Generation Roadmap (2026-06-07):
  [issue_2468_adversarial_generation_roadmap.md](issue_2468_adversarial_generation_roadmap.md)
* Issue #3474 Seed-Overlap Policy For Held-Out Adversarial Evidence (2026-06-23):
  [issue_3474_seed_overlap_policy.md](issue_3474_seed_overlap_policy.md)
* Issue #2524 Adversarial Scenario Manifest Generation (2026-06-07):
  [issue_2524_adversarial_manifests.md](issue_2524_adversarial_manifests.md)
* Issue #2562 Adversarial Manifest Planner Smoke (2026-06-07):
  [issue_2562_adversarial_manifest_smoke.md](issue_2562_adversarial_manifest_smoke.md)
* Issue #2567 Adversarial Manifest Quality Metrics (2026-06-07):
  [issue_2567_adversarial_manifest_quality.md](issue_2567_adversarial_manifest_quality.md)
* Issue #2568 Adversarial Expansion Gate (2026-06-07):
  [issue_2568_adversarial_expansion_gate.md](issue_2568_adversarial_expansion_gate.md)
* Issue #2618 Generated Adversarial Manifest Planner Smoke (2026-06-11):
  [issue_2618_adversarial_manifest_smoke.md](issue_2618_adversarial_manifest_smoke.md)
* Issue #2658 Validator-Runner Adversarial Manifest Smoke (2026-06-12):
  [issue_2658_adversarial_manifest_smoke.md](issue_2658_adversarial_manifest_smoke.md)
* Issue #2725 Generator-Readiness Contract (2026-06-13):
  [issue_2725_generator_readiness.md](issue_2725_generator_readiness.md)
* Issue #1457 Adversarial Map And Start-State Generation Protocol (2026-05-23):
  [issue_1457_adversarial_generation_protocol.md](issue_1457_adversarial_generation_protocol.md)
* Issue #1500 Adversarial Campaign Manifest Freeze (2026-05-26):
  [issue_1500_adversarial_manifest.md](issue_1500_adversarial_manifest.md)
* Issue #1571 Adversarial Smoke Packet Sharpening (2026-05-27):
  [issue_1571_adversarial_smoke_packet_sharpening.md](issue_1571_adversarial_smoke_packet_sharpening.md)
* Issue #1502 Adversarial Two-Family Run (2026-05-31):
  [issue_1502_adversarial_two_family_run.md](issue_1502_adversarial_two_family_run.md)
* Issue #1878 Head-On Route Replay Determinism (2026-05-31):
  [issue_1878_head_on_route_replay_determinism.md](issue_1878_head_on_route_replay_determinism.md)
* Issue #1503 Adversarial Stress-Coverage Synthesis (2026-05-31):
  [issue_1503_adversarial_stress_synthesis.md](issue_1503_adversarial_stress_synthesis.md)
* Issue #1963 Adversarial Parent Closeout (2026-06-01):
  [issue_1963_adversarial_parent_closeout.md](issue_1963_adversarial_parent_closeout.md)
* Issue #1861 Adversarial Replay Determinism Gate (2026-05-31):
  [issue_1861_adversarial_replay_determinism_gate.md](issue_1861_adversarial_replay_determinism_gate.md)
* Issue #1904 Scenario Perturbation Criticality Pilot (2026-05-31):
  [issue_1904_scenario_perturbation_criticality_pilot.md](issue_1904_scenario_perturbation_criticality_pilot.md)
* Issue #1933 Perturbation Seed Coverage (2026-05-31):
  [issue_1933_perturbation_seed_coverage.md](issue_1933_perturbation_seed_coverage.md)
* Issue #1935 Stronger Perturbation Planner (2026-05-31):
  [issue_1935_stronger_perturbation_planner.md](issue_1935_stronger_perturbation_planner.md)
* Issue #1937 Pedestrian Route Offset Pilot (2026-05-31):
  [issue_1937_ped_route_offset.md](issue_1937_ped_route_offset.md)
* Issue #1939 Corridor Trace Response (2026-05-31):
  [issue_1939_corridor_trace_response.md](issue_1939_corridor_trace_response.md)
* Issue #1941 Pedestrian Timing Phase Perturbation (2026-05-31):
  [issue_1941_ped_timing_phase.md](issue_1941_ped_timing_phase.md)
* Issue #1943 Single-Pedestrian Speed Perturbation (2026-05-31):
  [issue_1943_ped_speed_perturbation.md](issue_1943_ped_speed_perturbation.md)
* Issue #1945 ORCA Leave-Group Speed Trace (2026-06-01):
  [issue_1945_orca_leave_group_speed_trace.md](issue_1945_orca_leave_group_speed_trace.md)
* Issue #1947 Intersection-Wait Timing Vs Speed Trace (2026-06-01):
  [issue_1947_intersection_wait_timing_speed_trace.md](issue_1947_intersection_wait_timing_speed_trace.md)
* Issue #1949 Pedestrian Wait-Duration Perturbation (2026-06-01):
  [issue_1949_ped_wait_duration_perturbation.md](issue_1949_ped_wait_duration_perturbation.md)
* Issue #1951 Intersection-Wait Phase Grid (2026-06-01):
  [issue_1951_intersection_wait_phase_grid.md](issue_1951_intersection_wait_phase_grid.md)
* Issue #1953 Intersection-Wait Speed-Grid Trace (2026-06-01):
  [issue_1953_intersection_wait_speed_grid_trace.md](issue_1953_intersection_wait_speed_grid_trace.md)
* Issue #1965 Perturbation Criticality Synthesis (2026-06-01):
  [issue_1965_perturbation_criticality_synthesis.md](issue_1965_perturbation_criticality_synthesis.md)
* Issue #2222 Perturbation Criticality Metric (2026-06-04):
  [issue_2222_perturbation_criticality_metric.md](issue_2222_perturbation_criticality_metric.md)
* Issue #1304 pedestrian config boundary:
  [issue_1304_pedestrian_config_boundary.md](issue_1304_pedestrian_config_boundary.md)
* Issue #1633 RobotEnv SNQI proxy extraction:
  [issue_1633_robot_env_snqi_proxy.md](issue_1633_robot_env_snqi_proxy.md)
* Issue #1342 GH-Act Runtime Requirements:
  [issue_1342_gh_act_runtime_requirements.md](issue_1342_gh_act_runtime_requirements.md)
* Issue #1387 Tentabot-style value scorer spike:
  [issue_1387_tentabot_value_scorer_spike.md](issue_1387_tentabot_value_scorer_spike.md)
  (includes the Issue #1832 progress-recovery and Issue #1877 static-gate probe boundaries)
* Issue #1344 paired AMV primary protocol report:
  [issue_1344_paired_amv_protocol_report.md](issue_1344_paired_amv_protocol_report.md)
* SLURM issue batch status 2026-05-21 (canonical SLURM issue-status ledger):
  [slurm_issue_batch_status_2026-05-21.md](slurm_issue_batch_status_2026-05-21.md)
* SLURM job discovery snapshot 2026-05-31:
  [slurm_job_discovery_2026-05-31.md](slurm_job_discovery_2026-05-31.md)
* Issue #1776 State-Label Routing:
  [issue_1776_state_label_routing.md](issue_1776_state_label_routing.md)
* Issue #1397 Oracle Imitation Launch Packet:
  [issue_1397_oracle_imitation_launch_packet.md](issue_1397_oracle_imitation_launch_packet.md)
* Issue #1353 Broader AMV Baseline Preflight:
  [issue_1353_broader_amv_preflight.md](issue_1353_broader_amv_preflight.md)
* Issue #1484 Broader Cross-Kinematics Launch Packet:
  [issue_1484_broader_cross_kinematics_launch.md](issue_1484_broader_cross_kinematics_launch.md)
* Issue #1546 AMV actuation-envelope stress slice:
  [issue_1546_amv_actuation_envelope_stress_slice.md](issue_1546_amv_actuation_envelope_stress_slice.md)
* Issue #1556 synthetic AMV actuation stress slice and #1570 claim boundary:
  [issue_1556_amv_actuation_stress_slice.md](issue_1556_amv_actuation_stress_slice.md)
* Issue #2230 AMV actuation evidence ladder:
  [issue_2230_amv_actuation_evidence_ladder.md](issue_2230_amv_actuation_evidence_ladder.md)
* Issue #2011 AMV actuation-envelope sensitivity pilot:
  [issue_2011_amv_actuation_sensitivity_sweep.md](issue_2011_amv_actuation_sensitivity_sweep.md)
* Issue #1606 Full Classic placeholder retirement:
  [issue_1606_full_classic_placeholder_retirement.md](issue_1606_full_classic_placeholder_retirement.md)
* Issue #1744 Latency Stress Preflight Contract (2026-05-30):
  [issue_1744_latency_stress_preflight_contract.md](issue_1744_latency_stress_preflight_contract.md)
* Issue #1629 Latency-Aware Learned Navigation Safety (2026-05-30):
  [issue_1629_latency_aware_learned_navigation.md](issue_1629_latency_aware_learned_navigation.md)
* Issue #1398 metric rollup reconciliation:
  [issue_1398_metric_rollup_reconciliation.md](issue_1398_metric_rollup_reconciliation.md)
* Issue #1396 Shielded PPO Repair Launch Packet:
  [issue_1396_shielded_ppo_launch_packet.md](issue_1396_shielded_ppo_launch_packet.md)
* Issue #1474 Shielded PPO Repair Closeout (2026-06-01):
  [issue_1474_shielded_ppo_repair_closeout.md](issue_1474_shielded_ppo_repair_closeout.md)
* Issue #2006 Guarded-PPO Zero-Motion Repair (2026-06-01):
  [issue_2006_guarded_ppo_zero_motion_repair.md](issue_2006_guarded_ppo_zero_motion_repair.md)
* Issue #2008 Artifact Catalog Contract (2026-06-01):
  [issue_2008_artifact_catalog.md](issue_2008_artifact_catalog.md)
* Issue #2037 Artifact Compiler Smoke (2026-06-01):
  [issue_2037_artifact_compiler_smoke.md](issue_2037_artifact_compiler_smoke.md)
* Issue #2040 Artifact Publication Workflow (2026-06-01):
  [issue_2040_artifact_publication_workflow.md](issue_2040_artifact_publication_workflow.md)
* Issue #2460 Evidence Bundle v1:
  [issue_2460_evidence_bundle_v1.md](issue_2460_evidence_bundle_v1.md)
* Issue #2034 Platformization Roadmap (2026-06-01):
  [issue_2034_platformization_roadmap.md](issue_2034_platformization_roadmap.md)
* Issue #1395 Learned Risk Model Launch Packet:
  [issue_1395_learned_risk_launch_packet.md](issue_1395_learned_risk_launch_packet.md)
* Issue #2273 learned-risk trace manifest preflight:
  [issue_2273_learned_risk_trace_preflight.md](issue_2273_learned_risk_trace_preflight.md)
* Issue #1686 Learned-Policy Artifact Manifest Fields (2026-05-30):
  [artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md) and
  [open_issues_training_split_audit_2026-05-30.md](open_issues_training_split_audit_2026-05-30.md)
* Issue #1966 ScenarioBelief Interface Design (2026-06-01):
  [issue_1966_scenario_belief_interface.md](issue_1966_scenario_belief_interface.md)
* Issue #2477 ScenarioBelief Contract Bridge (2026-06-06):
  [issue_2477_scenario_belief_contract.md](issue_2477_scenario_belief_contract.md)
* Issue #2478 Uncertainty-Aware ScenarioBelief Contract (2026-06-06):
  [issue_2478_uncertainty_scenario_belief.md](issue_2478_uncertainty_scenario_belief.md)
* Issue #2528 ScenarioBelief Consumer Smoke (2026-06-07):
  [issue_2528_scenario_belief_consumer_smoke.md](issue_2528_scenario_belief_consumer_smoke.md)
* Issue #2565 Uncertainty-Aware Stream-Gap Gating Smoke (2026-06-07):
  [issue_2565_uncertainty_gating_smoke.md](issue_2565_uncertainty_gating_smoke.md)
* Issue #2538 ScenarioBelief Planner Projection Smoke (2026-06-07):
  [issue_2538_scenario_belief_planner_projection.md](issue_2538_scenario_belief_planner_projection.md)
* Issue #2479 Real-Trajectory Scenario Prior Scope (2026-06-06):
  [issue_2479_real_trajectory_scenario_prior.md](issue_2479_real_trajectory_scenario_prior.md)
* Issue #2523 Scenario-Prior Smoke:
  [issue_2523_scenario_prior_smoke.md](issue_2523_scenario_prior_smoke.md)
  ([summary](evidence/issue_2523_scenario_prior_smoke/summary.json),
  [artifact](evidence/issue_2523_scenario_prior_smoke/scenario_prior.v1.json))
* Issue #2917 ScenarioPrior.v1 Provenance Cards (2026-06-19):
  [issue_2917_scenario_prior_cards.md](issue_2917_scenario_prior_cards.md)
  ([registry](../../configs/research/scenario_prior_cards_issue_2917.yaml))
* Issue #2474 Signalized Pedestrian Crossing Benchmark Scope (2026-06-06):
  [issue_2474_signalized_crossing_benchmark.md](issue_2474_signalized_crossing_benchmark.md)
* Issue #2662 Signal-State Promotion Contract (2026-06-12):
  [issue_2662_signal_state_promotion_contract.md](issue_2662_signal_state_promotion_contract.md)
* Issue #2799 Signalized Crossing Runtime Evidence (2026-06-13):
  [evidence/issue_2799_signalized_runtime/README.md](evidence/issue_2799_signalized_runtime/README.md)
  ([summary](evidence/issue_2799_signalized_runtime/summary.json))
* Issue #2475 Probabilistic Prediction Interface (2026-06-06):
  [issue_2475_probabilistic_prediction_interface.md](issue_2475_probabilistic_prediction_interface.md)
* Issue #1638 local model path preflight:
  [issue_1638_model_path_preflight.md](issue_1638_model_path_preflight.md)
* Issue #1960 local artifact retirement status:
  [issue_1960_local_artifact_retirement.md](issue_1960_local_artifact_retirement.md)
* Issue #2277 local artifact classification:
  [issue_2277_local_artifact_classification.md](issue_2277_local_artifact_classification.md)
* Issue #2313 local baseline quarantine:
  [issue_2313_local_baseline_quarantine.md](issue_2313_local_baseline_quarantine.md)
* Issue #2409 local baseline quarantine decision:
  [issue_2409_local_baseline_quarantine_decision.md](issue_2409_local_baseline_quarantine_decision.md)
* Issue #1845 report grouping contracts:
  [issue_1845_report_grouping_contracts.md](issue_1845_report_grouping_contracts.md)
* Learned local-navigation policy registry:
  [policy_search/learned_policy_registry.md](policy_search/learned_policy_registry.md)
* Issue #1758 Arena-Rosnav Source-Side Assessment (2026-05-30):
  [policy_search/issue_1758_arena_rosnav_source_assessment.md](policy_search/issue_1758_arena_rosnav_source_assessment.md)
* Issue #1620 External Learned Local-Navigation Ranking (2026-05-30):
  [policy_search/2026-05-30_external_learned_policy_ranking_issue_1620.md](policy_search/2026-05-30_external_learned_policy_ranking_issue_1620.md)
* Issue #1626 foundation-model local-navigation readiness (2026-05-30):
  [policy_search/2026-05-30_foundation_model_readiness_issue_1626.md](policy_search/2026-05-30_foundation_model_readiness_issue_1626.md)
* Issue #1621 diffusion-policy local-navigation feasibility (2026-05-30):
  [policy_search/2026-05-30_diffusion_policy_feasibility_issue_1621.md](policy_search/2026-05-30_diffusion_policy_feasibility_issue_1621.md)
* Issue #1615 LiDAR Learned-Policy Launch Plan (2026-05-29):
  [issue_1615_lidar_learned_policy_plan.md](issue_1615_lidar_learned_policy_plan.md)
* Issue #1618 learned local-policy adapter interface:
  [issue_1618_learned_policy_adapter_interface.md](issue_1618_learned_policy_adapter_interface.md)
* Issue #1627 learned-policy transfer benchmark design:
  [issue_1627_learned_policy_transfer_benchmark.md](issue_1627_learned_policy_transfer_benchmark.md)
* Issue #1761 learned-policy transfer metadata validator:
  [issue_1761_learned_policy_transfer_metadata_validator.md](issue_1761_learned_policy_transfer_metadata_validator.md)
* Issue #1677 SiT Dataset Terms Audit (2026-05-29):
  [issue_1677_sit_dataset_terms.md](issue_1677_sit_dataset_terms.md)
* Issue #1689 Simulation Trace Export Schema (2026-05-30):
  [issue_1689_simulation_trace_export_schema.md](issue_1689_simulation_trace_export_schema.md)
* Issue #2038 Real Trace Viewer Smoke (2026-06-01):
  [issue_2038_real_trace_viewer_smoke.md](issue_2038_real_trace_viewer_smoke.md)
* Issue #2236 Trace Mechanism Evidence Rubric (2026-06-04):
  [issue_2236_trace_mechanism_evidence_rubric.md](issue_2236_trace_mechanism_evidence_rubric.md)
* Issue #2463 Mechanism Signal Checker (2026-06-06):
  [issue_2463_mechanism_signal_checker.md](issue_2463_mechanism_signal_checker.md)
* Issue #2263 Mechanism Activation Report Fields (2026-06-05):
  [issue_2263_mechanism_activation_report_fields.md](issue_2263_mechanism_activation_report_fields.md)
* Issue #2227 Mechanism Panel Readiness (2026-06-04):
  [issue_2227_mechanism_panels.md](issue_2227_mechanism_panels.md)
* Issue #2428 AMMV Mechanism Trace Panels (2026-06-06):
  [issue_2428_mechanism_trace_panels.md](issue_2428_mechanism_trace_panels.md)
* Issue #2923 Mechanism Trace v1 Schema (2026-06-16):
  [issue_2923_mechanism_trace_schema.md](issue_2923_mechanism_trace_schema.md)
* Issue #2667 Trace Failure Predicate Tables (2026-06-12):
  [issue_2667_trace_failure_predicate_tables.md](issue_2667_trace_failure_predicate_tables.md)
* Issue #2688 Trace Predicate Benchmark Matrix (2026-06-15):
  [issue_2688_trace_predicate_matrix.md](issue_2688_trace_predicate_matrix.md)
* Issue #2270 Mechanism Panel Candidate Manifest (2026-06-05):
  [issue_2270_panel_candidate_manifest.md](issue_2270_panel_candidate_manifest.md)
* Issue #1676 Proxemic Profile Comfort Slice (2026-05-30):
  [issue_1676_proxemic_profile_comfort_slice.md](issue_1676_proxemic_profile_comfort_slice.md)
* Issue #1617 Local-Planner Repository Survey (2026-05-29):
  [issue_1617_local_planner_repo_survey.md](issue_1617_local_planner_repo_survey.md)
* Issue #1662 LiDAR PPO MLP Smoke (2026-05-29):
  [issue_1662_lidar_ppo_smoke.md](issue_1662_lidar_ppo_smoke.md)
* Issue #1294 seed-sensitivity perturbations:
  [issue_1294_seed_sensitivity_perturbations.md](issue_1294_seed_sensitivity_perturbations.md)
* Issue #1608 scenario seed sensitivity:
  [issue_1608_seed_sensitivity_analysis.md](issue_1608_seed_sensitivity_analysis.md)
* Issue #1609 seed-sensitive scenario mechanisms:
  [issue_1609_seed_sensitive_mechanisms.md](issue_1609_seed_sensitive_mechanisms.md)
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
* Issue #1692 topology-hypothesis trace probe:
  [issue_1692_topology_hypothesis_probe.md](issue_1692_topology_hypothesis_probe.md)
* Issue #2223 topology-hypothesis planning diagnostic:
  [issue_2223_topology_hypothesis_planning.md](issue_2223_topology_hypothesis_planning.md)
* Issue #2258 Topology Primary-Route Audit 2026-06-05:
  [issue_2258_topology_primary_route_audit.md](issue_2258_topology_primary_route_audit.md)
* Issue #2307 topology score diagnostic:
  [issue_2307_topology_score_diagnostic.md](issue_2307_topology_score_diagnostic.md)
* Issue #2393 topology selection preflight:
  [issue_2393_topology_selection_preflight.md](issue_2393_topology_selection_preflight.md)
* Issue #2403 topology selection-score decision:
  [issue_2403_topology_selection_score_decision.md](issue_2403_topology_selection_score_decision.md)
* Issue #2518 topology near-parity gate diagnostic:
  [issue_2518_topology_near_parity_gate.md](issue_2518_topology_near_parity_gate.md)
* Issue #2530 topology near-parity corrective-behavior smoke:
  [issue_2530_topology_near_parity_corrective_smoke.md](issue_2530_topology_near_parity_corrective_smoke.md)
* Issue #2563 Topology Corrective Revision Proposal 2026-06-07:
  [issue_2563_topology_corrective_revision.md](issue_2563_topology_corrective_revision.md)
* Issue #2600 Topology Revision Decision 2026-06-11:
  [issue_2600_topology_revision_decision.md](issue_2600_topology_revision_decision.md)
* Issue #2621 Topology Revision Hypothesis 2026-06-11:
  [issue_2621_topology_revision_hypothesis.md](issue_2621_topology_revision_hypothesis.md)
* Issue #2624 Topology Reuse-Penalty Paired Diagnostic Gate 2026-06-11:
  [issue_2624_topology_reuse_penalty_gate.md](issue_2624_topology_reuse_penalty_gate.md)
  ([summary](evidence/issue_2624_topology_reuse_penalty_gate/summary.json))
* Issue #2660 Topology Successor Gate After Reuse-Penalty Regression (superseded by Issue #2704):
  [issue_2660_topology_successor_gate.md](archive/issue_2660_topology_successor_gate.md)
* Issue #2704 Progress-Gated Topology Successor Diagnostic:
  [issue_2704_progress_gated_topology_successor.md](issue_2704_progress_gated_topology_successor.md)
  ([summary](evidence/issue_2704_progress_gated_topology_successor/summary.json))
* Issue #2716 topology reselection cross-slice diagnostic:
  [issue_2716_topology_reselection_cross_slice.md](issue_2716_topology_reselection_cross_slice.md)
  ([summary](evidence/issue_2716_topology_reselection_cross_slice/summary.json))
* Issue #2742 topology reselection successor launch packet:
  [issue_2742_topology_reselection_successor.md](issue_2742_topology_reselection_successor.md)
  ([summary](evidence/issue_2742_topology_reselection_successor/summary.json))
* Issue #2752 topology reselection mechanism diagnosis:
  [issue_2752_topology_reselection_mechanism.md](issue_2752_topology_reselection_mechanism.md)
  ([evidence](evidence/issue_2752_topology_reselection_mechanism/README.md))
* Issue #2706 topology lane synthesis after progress-gated successor:
  [issue_2706_topology_lane_synthesis.md](issue_2706_topology_lane_synthesis.md)
* Issue #2522 topology and AMV why-first diagnostics:
  [issue_2522_why_first_diagnostics.md](issue_2522_why_first_diagnostics.md)
  ([topology report](evidence/issue_2522_why_first_diagnostics/topology_near_parity_why_first_report.md),
  [AMV report](evidence/issue_2522_why_first_diagnostics/amv_actuation_why_first_report.md))
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
  artifacts out of `output/` into git. [Issue #2460 Evidence Bundle v1](issue_2460_evidence_bundle_v1.md)
  defines the explicit-file helper and schema contract. Current bundles include the
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
  [Issue #852 queue-fill single-factor batch evidence](evidence/issue_852_queue_fill_batch_2026-06-06/README.md),
  [Issue #2428 AMMV mechanism trace panels](evidence/issue_2428_mechanism_trace_panels_2026-06-06/README.md),
  [Issue #2432 AMMV trace selection evidence](evidence/issue_2432_ammv_trace_selection_2026-06-06/README.md),
  [Issue #2434 AMMV scenario sweep evidence](evidence/issue_2434_ammv_scenario_sweep_2026-06-06/README.md),
  [Issue #3207 Fidelity Sensitivity Launch Packet 2026-06-20](evidence/issue_3207_fidelity_sensitivity_launch_packet_2026-06-20/README.md),
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
* [Issue #2458 Human-Interaction Proxy Metrics](issue_2458_human_interaction_proxy_metrics.md)
  defines the schema-backed `human-interaction-proxy.v1` block, simulation-proxy formulas, and
  claim boundary for human-centered mechanism-report reductions.
* [Issue #3184 Distributional Disruption Metric Contract](issue_3184_distributional_disruption_metrics.md)
  defines observable simulation-state cohorts, candidate distributional disruption formulas,
  prohibited terminology, non-claims, and the evidence gate before any implementation PR.
  The [Issue #3206 heterogeneous-pedestrian smoke evidence](evidence/issue_3206_heterogeneous_pedestrian_smoke_2026-06-20/README.md)
  records the #3261 decision that speed-tier cohorts are the current archetype-adjacent surface and
  named archetype cohorts remain deferred.
* [Issue #2456 AMV Local Navigation Evaluation Suite (proposal)](issue_2456_amv_local_nav_evaluation_suite.md)
  defines the proposed AMV local navigation suite structure with eight evaluation dimensions,
  evidence-boundary classification, and related issue links.
* [Issue #2452 Mechanism-Aware Local-Navigation Suites (proposal)](issue_2452_mechanism_aware_local_nav_suites.md)
  defines seven proposed local-navigation suite contracts that group existing scenario IDs by
  failure mechanism, required traces, metrics, and claim boundaries.
* [Issue #2453 Planner Mechanism Cards](issue_2453_planner_mechanism_cards.md)
  records mechanism, activation, positive/negative evidence, transfer status, and next-proof
  boundaries for active planner research candidates.
* [Issue #2443 AMV Actuation Trace Review](issue_2443_amv_trace_review.md)
  records the compact progress-vs-clipping review for the matched AMV actuation-smoke pair and
  fails closed on missing raw frame/event IDs.
* [Issue #2446 AMV Actuation Feasibility Ranking](issue_2446_amv_feasibility_ranking.md)
  classifies actuation feasibility as a diagnostic secondary ranking signal for the matched AMV
  actuation-smoke pair without upgrading it to planner-improvement or benchmark evidence.
* [Issue #3170 AMV Feasibility Ranking Stress Synthesis](issue_3170_amv_feasibility_ranking_stress.md)
  compares that one-scenario actuation-aware signal against the broader tracked AMMV/default
  multi-scenario slice, finds the broader slice frame-identical, and keeps the ranking claim
  diagnostic-only pending direct actuation-aware multi-scenario evidence.
* [Issue #2440 AMV Timeout Closure](issue_2440_amv_timeout_closure.md)
  closes the timeout-driver classification as `feasibility_improved_but_route_blocked` and keeps
  actuation-aware scoring diagnostic-only for planner improvement.
* [Issue #2522 Why-First Diagnostics](issue_2522_why_first_diagnostics.md)
  publishes generated why-first reports for the topology near-parity and AMV actuation timeout
  diagnostics while preserving both lanes at `revise`.
* [Issue #1092 Multi-AMV First Slice](issue_1092_multi_amv_first_slice.md)
  records the minimal multi-robot scenario surface, smoke runner, inter-robot metrics, and deferred
  fleet-integration boundary.
* [Issue #1128 Multi-AMV Episode Extension](issue_1128_multi_amv_episode_extension.md)
  records the canonical `metrics.inter_robot` JSONL/report output contract for the explicit
  multi-AMV smoke path.
* [Issue #1168 Multi-AMV Planner Support Classification](issue_1168_multi_amv_planner_support.md)
  records the current planner-family inventory, fail-closed support gate, and the boundary between
  goal-controller smoke execution and real multi-AMV planner support.
* [Issue #1660 LiDAR Tracked-Agent Adapter](issue_1660_lidar_tracked_agent_adapter.md)
  records the testing-only `lidar_social_force` adapter contract, endpoint-cluster tracking
  assumptions, and fail-closed boundary for LiDAR-derived social-state planner inputs.
* [Issue #1659 LiDAR Ego Occupancy Adapter](issue_1659_lidar_ego_occupancy_adapter.md)
  records the testing-only `lidar_grid_route` adapter contract, opt-in gate, and fail-closed
  boundary between LiDAR-derived ego occupancy and privileged map/SocNav state.
* [Issue #1091 SDD Importer](issue_1091_sdd_importer.md)
  records the one-dataset-first real-world trajectory import boundary, SDD license assumptions,
  importer outputs, and deferred generalization scope.
* [Issue #1090 Observation Visibility](issue_1090_observation_visibility.md)
  records the planner-facing FOV/range/static-occlusion boundary, ground-truth separation, and
  dynamic-occlusion follow-up boundary.
* [Issue #1108 BC Warm-Start PPO Execution](issue_1108_bc_warm_start_execution.md)
  records the imitation observation-contract blocker, unblock patch, one-episode real collection
  preflight, and Slurm job IDs for the #749 BC-preinitialized PPO chain.
* [Issue #1961 BC Warm-Start Recoverability](issue_1961_bc_warm_start_recoverability.md)
  classifies the #1108 artifact trail as `rerun_required`: dataset and BC checkpoint are preserved
  in W&B, but final PPO and comparison evidence are missing.
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
  records the first post-MVP steering-mode bundle, artifact-filterability contract, and issue #1604
  `ego_up_view_v1` renderer-hook implementation proof.

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
* [Issue #1752 Decision Transformer Dataset Preflight](issue_1752_decision_transformer_dataset_preflight.md)
  records the reward/terminal/return-to-go trajectory schema, manifest metadata, validator CLI, and
  tiny dry-run proof needed before any future Decision Transformer training campaign.
* [Issue #1209 Imitation Observation Contract](issue_1209_imitation_observation_contract.md)
  records the BR-06 checkpoint-compatible observation-contract fix and validation path that
  unblocks #1108's BC warm-start launch.
* [Issue #1024 H500 PPO Retrain](issue_1024_h500_ppo_retrain.md)
  records the all-available scenario surface, PR #1025 h500 horizon alignment, and SLURM job
  `12350` for the first 12M-step PPO retrain.
* [Issue #2557 Recovered Reward-Curriculum Seed Runs](issue_2557_recovered_diagnostic_seeds.md)
  records three reward-curriculum seed runs (506/508/509) whose manifests were lost to the
  cross-worktree serializer bug and backfilled via #3590; diagnostic-tier only (marginal SNQI,
  elevated collision), with an in-flight 501–511 variance fill.

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
* [Issue #2214 Hot-Path Optimization Synthesis](./issue_2214_hot_path_synthesis.md)
  records the 2026-06-04 diagnostic comparison for the simulator hot-path optimization wave and
  classifies the local smoke evidence as startup dominated rather than a broad speedup claim.
* [Issue #3025 Large Crowd Step Profiling](./issue_3025_large_crowd_step_profile.md)
  documents the merged `--large-crowd-profile`/`--step-profile` harness changes, explicit cold-start
  versus steady-mode warmup behavior, and a constrained diagnostic-to-optimization boundary for
  future action.

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
* [Issue #2620 Oracle-Imitation Artifact Access Audit](issue_2620_oracle_artifact_access.md)
  records that the #2441 split/checksum evidence remains useful, but raw trace retrieval is blocked
  because durable pointers are missing and the recorded local `output/slurm` paths are absent.
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
* [Issue #1751 Policy Stack Arbitration Trace Packet](issue_1751_policy_stack_arbitration_trace.md)
  defines the diagnostic-only `policy_stack_v1.arbitration_trace_packet.v1` contract for future
  learned-arbiter data collection, plus the compact validation smoke command.
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
* [Issue #1856 Predictive-v2 Coupling Objective](issue_1856_predictive_coupling_objective.md)
  records the local closed-loop gate and planner-side coupling hypothesis that should run before
  any renewed predictive-v2 four-way expansion.
* [Issue #1897 Predictive Coupling Gate Preflight](issue_1897_predictive_coupling_gate_preflight.md)
  records the local closed-loop gate execution, its fail-closed result, and why the old predictive-v2
  four-way expansion should remain blocked.
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
- [Issue #2015 MuJoCo AMV Micro-Backend Diagnostic](issue_2015_mujoco_amv_micro_backend.md)
  records the optional diagnostic-only MuJoCo micro-backend probe for AMV command-trace actuation
  response, with no routine dependency or calibrated/benchmark claim.

## DreamerV3 Notes

* [DreamerV3 Program Full Handoff (2026-04-28)](dreamerv3_program_full_handoff_2026_04_28.md)
  Consolidated execution plan for issues #578, #608, #609, #782, and #789.
* [DreamerV3 BR-08 Full Progress (2026-04-29)](dreamerv3_br08_full_progress_2026_04_29.md)
  Run-level outcome and diagnostics summary for Slurm 12159.
* [DreamerV3 Program Close-Out (2026-04-30)](dreamerv3_program_close_out_2026_04_30.md)
  Program-level stop decision and closure rationale after the probe/gate/full sequence.
* [Issue #1623 World-Model Navigation Feasibility](issue_1623_world_model_feasibility.md)
  Decision note for DreamerV3, PlaNet, TD-MPC2, and DreamerNav-style candidates: monitor external
  methods, reject another local flat-vector retrain, and require source/provenance preflights first.
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
* [Issue #2274 Hybrid-Learning Component Evidence Status Matrix](issue_2274_hybrid_component_matrix.md)
  classifies learned risk, ORCA-residual BC, oracle imitation, shielded PPO repair, and BC
  warm-start PPO before Issue #1489 synthesis. It records blockers, next actions, and the
  conclusion that Issue #1489 remains blocked from comparative synthesis.
* [Issue #2410 Hybrid-Learning Component Readiness Refresh](issue_2410_hybrid_component_readiness_refresh.md)
  refreshes the component matrix after the ORCA-residual v1 progress-probe launch-packet revision
  and keeps Issue #1489 blocked from comparative synthesis.

* [Issue #2408 ORCA-Residual Low-Progress Analysis](issue_2408_orca_residual_low_progress_analysis.md)
  classifies the failed v0 smoke row and recommends only a revised, instrumented progress-probe
  rerun before any nominal escalation.

* [Issue #2411 Predictive-v2 Child Classification](issue_2411_predictive_v2_child_classification.md)
  classifies stale predictive-v2 child issues after the stop-old-expansion decision so blocked
  proposal work is not treated as benchmark-ready execution.

* [Issue #3254 Predictive Crossing-Conflict Negative Result](issue_3254_predictive_crossing_conflict_negative_result.md)
  preserves the schema-fixed crossing-conflict predictive retraining outcome: training completed,
  but final evaluation failed the success-rate gate, so the run is not planner promotion evidence.
* [Issue #3985 ACMPC Feasibility Assessment](issue_3985_acmpc_feasibility_assessment.md)
  records the assessment-only boundary for an Actor-Critic Model Predictive Control inspired
  learned-MPC local planner, including adapter burden, benchmark claim limits, and a conditional
  design-child recommendation.
