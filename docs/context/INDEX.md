# Context Retrieval Index

Publication figure style pack (opt-in vector export, colorblind-safe planner palette, provenance sidecars, LaTeX-safe captions):
[issue_4777_publication_figure_style_pack.md](issue_4777_publication_figure_style_pack.md).

Recent OMPL ST-RRT* feasibility assessment:
[issue_4797_ompl_strrtstar_assessment.md](issue_4797_ompl_strrtstar_assessment.md).

Recent SocNavBench ETH stage/SVG closure audit and generated SVG evidence:
[issue_4546_closure_audit_2026-07-06.md](issue_4546_closure_audit_2026-07-06.md).

Recent Simple Linux Utility Resource Management (SLURM)-to-claim closure audit:
[issue_3425_closure_audit_2026-07-05.md](evidence/issue_3425_closure_audit_2026-07-05.md).

Recent campaign fail-fast contract closure audit:
[issue_4684_closure_audit_2026-07-07.md](evidence/issue_4684_closure_audit_2026-07-07.md).

Recent topology corrective-behaviors closure audit:
[issue_3463_closure_audit_2026-07-07.md](evidence/issue_3463_closure_audit_2026-07-07.md).

High-churn topology corrective-behaviors issue state:
[issue_3463_state.yaml](issue_3463_state.yaml).

Recent Package C prediction/observation closure audit:
[issue_3080_closure_audit_2026-07-05.md](evidence/issue_3080_closure_audit_2026-07-05.md).

Recent uncertainty-representation generalization closure audit:
[closure_audit.md](evidence/issue_3557_uncertainty_representation_generalization/closure_audit.md).

Recent uncertainty-source generalization diagnostic evidence:
[README.md](evidence/issue_3557_uncertainty_source_generalization/README.md).

Recent reactivity-vs-replay rank-study closure audit:
[issue_3637_closure_audit_2026-07-05.md](evidence/issue_3637_closure_audit_2026-07-05.md).

Recent serial benchmark GPU memory closure audit:
[issue_4520_gpu_memory_closure_audit_2026-07-05.md](issue_4520_gpu_memory_closure_audit_2026-07-05.md).

Recent hard-guarded hybrid-learning synthesis closure audit:
[issue_1489_closure_audit_2026-07-05.md](evidence/issue_1489_closure_audit_2026-07-05.md).

Post-#4642 hard-guarded hybrid-learning integration audit:
[issue_1489_post_4642_integration_audit_2026-07-07.md](evidence/issue_1489_post_4642_integration_audit_2026-07-07.md).

Recent ORCA-residual BC smoke/nominal lane closure audit (keep open; blocked on a
Slurm smoke rerun + nominal escalation, CPU-side criteria delivered):
[issue_1475_closure_audit_2026-07-06.md](evidence/issue_1475_closure_audit_2026-07-06.md).
Recent ORCA-residual parent closure audit (keep open; local handoff ready,
parent blocked on Issue #1475 durable smoke/nominal evidence):
[issue_1358_closure_audit_2026-07-07.md](issue_1358_closure_audit_2026-07-07.md).
High-churn state propagation: [issue_1358_state.yaml](issue_1358_state.yaml),
[issue_1475_state.yaml](issue_1475_state.yaml).

Recent camera-ready decomposition closure audit:
[issue_3385_closure_audit_2026_07_04.md](issue_3385_closure_audit_2026_07_04.md).
Recent predictive-v2 closure audit: [issue_1490_closure_audit.md](issue_1490_closure_audit.md).

Issue: [#1714](https://github.com/ll7/robot_sf_ll7/issues/1714)

This is the first stop for broad `docs/context/` retrieval. Use it to choose the smallest current
context surface before reading issue history in bulk.

## Retrieval Order

1. Start with this index when the task says "context", "history", "handoff", "what happened", or
   mentions a broad subsystem.
2. Read the domain entry points below, then follow only the linked notes that match the task.
3. Use [README.md](README.md) when creating or maintaining notes, because it defines the note
   workflow and full discoverability list.
4. Use [memory/MEMORY.md](../../memory/MEMORY.md) for stable cross-session facts that should outlive
   one issue or PR.
5. Treat `output/` as disposable local state. Durable evidence belongs in small tracked manifests
   under [evidence/](evidence/README.md) or in an external artifact store with a tracked pointer.

## Status Rules

- `Current`: use as an active source of truth.
- `Historical`: useful background, but check newer linked notes before relying on conclusions.
- `Superseded`: keep only for provenance; use the replacement note named near the top.
- `Evidence`: tracked compact proof or manifest; do not treat missing local `output/` files as
  durable dependencies.
- `Proposal`: planned or exploratory guidance; do not cite as completed implementation evidence.

When updating a note, mark stale or superseded claims in place instead of stacking contradictory
paragraphs.

Machine-readable sidecar: [catalog.yaml](catalog.yaml) records the curated entry points below with
`status` and `freshness` metadata. Validate it with
`uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml`
or the normal diff wrapper.

## Pruning And Refactoring Rules

- Prune, rewrite, rename, or delete stale notes more aggressively when the reason is unambiguous,
  the note no longer provides reusable value, and benchmark or paper-facing evidence pointers are
  preserved.
- If two notes cover the same decision, keep the newer canonical note current and mark the older
  note `Historical` or `Superseded` near the top with a pointer to the replacement.
- If a note is issue-local and no longer reusable, keep it linked from the relevant issue-specific
  entry only, or remove it from this index when the issue/PR record already preserves the needed
  provenance.
- If a note supports a benchmark or paper-facing claim, preserve the evidence pointer even when the
  prose is shortened.
- Add a note to this index only when it is a current domain entry point, a durable policy boundary,
  or a curated context-pack ingredient.

## Canonical Context Surfaces

| Area | Current entry points | Use for |
|---|---|---|
| Agent workflow | [goal_driven_agent_loops_2026-05-13.md](goal_driven_agent_loops_2026-05-13.md), [open_issue_execution_improvement_plan_2026-05-30.md](open_issue_execution_improvement_plan_2026-05-30.md), [issue_713_batch_first_issue_workflow.md](issue_713_batch_first_issue_workflow.md), [issue_1776_state_label_routing.md](issue_1776_state_label_routing.md), [skill_consolidation_audit_2026-06-20.md](skill_consolidation_audit_2026-06-20.md), [Agent Workflow Lessons memory](../../memory/workflows/2026-05-31_agent_workflow_lessons.md) | Issue-to-PR loops, research-result mode, queue exhaustion, batching, issue splitting, state-label routing, live-state checks, delegated-worker proof boundaries, skill consolidation candidates, and GitHub workflow policy. |
| PR workflow contracts | [issue_3472_pr_body_contracts.md](issue_3472_pr_body_contracts.md), [../code_review.md](../code_review.md) | Live PR body, follow-up disposition, and domain-aware approval CI guard for reviewability and evidence-validity-sensitive PRs. |
| Context architecture | This file, [../ai/context_packing.md](../ai/context_packing.md), [../ai/retrieval_deferral.md](../ai/retrieval_deferral.md), [issue_728_coding_agents_compatibility.md](issue_728_coding_agents_compatibility.md) | Context-pack decisions, optional external tools, Markdown-first retrieval, and cross-agent compatibility. |
| Research-engine guide | [../researchers_guide.md](../researchers_guide.md), [research_month_one_synthesis_2026-06.md](research_month_one_synthesis_2026-06.md) | How to define a research question, choose an evidence tier, author a campaign manifest, run validation, interpret evidence grades, and the epic #3057 month-one landed-vs-pending synthesis. Synthesis/docs only — no new benchmark claim. |
| Platformization roadmap | [issue_2034_platformization_roadmap.md](issue_2034_platformization_roadmap.md), [issue_2040_artifact_publication_workflow.md](issue_2040_artifact_publication_workflow.md), [issue_2038_real_trace_viewer_smoke.md](issue_2038_real_trace_viewer_smoke.md), [issue_2013_backend_adapter_contract.md](issue_2013_backend_adapter_contract.md), [issue_2015_mujoco_amv_micro_backend.md](issue_2015_mujoco_amv_micro_backend.md), [issue_1894_slurm_job_finalizer.md](issue_1894_slurm_job_finalizer.md), [root_layout_structured_migration_2026-06-01.md](root_layout_structured_migration_2026-06-01.md) | Connected platform layer after June 2026 tooling merges: artifact publication, trace review, backend contracts, optional diagnostic simulator probes, SLURM closeout, external data staging, root-layout cleanup, and next stabilization PRs. |
| Benchmark evidence policy | [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md), [artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md), [issue_3290_simulation_model_credibility_checklist.md](issue_3290_simulation_model_credibility_checklist.md), [issue_3292_rare_event_probability_plan.md](issue_3292_rare_event_probability_plan.md), [issue_3293_evidence_integration_contract_inventory.md](issue_3293_evidence_integration_contract_inventory.md), [issue_3062_campaign_manifest_flow.md](issue_3062_campaign_manifest_flow.md), [issue_3061_social_compliance_metric_contract.md](issue_3061_social_compliance_metric_contract.md), [issue_3184_distributional_disruption_metrics.md](issue_3184_distributional_disruption_metrics.md), [issue_3059_research_engine_suite_v0.md](issue_3059_research_engine_suite_v0.md), [issue_3000_map_public_smoke_provenance.md](issue_3000_map_public_smoke_provenance.md), [issue_1584_socnav_unavailable_row_policy.md](issue_1584_socnav_unavailable_row_policy.md), [issue_2397_socnavbench_control_status_2026-06-06.md](issue_2397_socnavbench_control_status_2026-06-06.md), [issue_1436_reproducibility_flaky_acceptance.md](issue_1436_reproducibility_flaky_acceptance.md), [issue_2012_failure_mechanism_classifier.md](issue_2012_failure_mechanism_classifier.md), [issue_2220_failure_mechanism_taxonomy.md](issue_2220_failure_mechanism_taxonomy.md), [issue_2544_mechanism_aware_suite_smoke.md](issue_2544_mechanism_aware_suite_smoke.md), [issue_2586_static_deadlock_trace_fields.md](issue_2586_static_deadlock_trace_fields.md), [issue_2588_static_deadlock_controlled_trace.md](issue_2588_static_deadlock_controlled_trace.md), [issue_2590_escape_recenter_static_deadlock_controlled_trace.md](issue_2590_escape_recenter_static_deadlock_controlled_trace.md), [issue_2592_static_deadlock_active_row_h500.md](issue_2592_static_deadlock_active_row_h500.md), [issue_2547_counterfactual_mechanism_taxonomy.md](issue_2547_counterfactual_mechanism_taxonomy.md), [issue_1272_validation_falsification_strategy.md](issue_1272_validation_falsification_strategy.md), [issue_2231_mechanism_aware_ranking.md](issue_2231_mechanism_aware_ranking.md), [mechanism_closure_status.md](mechanism_closure_status.md), [issue_2522_why_first_diagnostics.md](issue_2522_why_first_diagnostics.md), [issue_2531_amv_trace_boundary.md](issue_2531_amv_trace_boundary.md), [issue_2222_perturbation_criticality_metric.md](issue_2222_perturbation_criticality_metric.md), [issue_2234_predictive_perturbation_criticality.md](issue_2234_predictive_perturbation_criticality.md), [issue_2275_predictive_v2_fate.md](issue_2275_predictive_v2_fate.md), [issue_2230_amv_actuation_evidence_ladder.md](issue_2230_amv_actuation_evidence_ladder.md), [issue_2224_amv_actuation_ranking.md](issue_2224_amv_actuation_ranking.md), [issue_2268_amv_timeout_decomposition.md](issue_2268_amv_timeout_decomposition.md), [issue_2308_amv_timeout_trace_analysis.md](issue_2308_amv_timeout_trace_analysis.md), [issue_2404_amv_timeout_decomposition_decision.md](issue_2404_amv_timeout_decomposition_decision.md), [issue_2440_amv_timeout_closure.md](issue_2440_amv_timeout_closure.md), [issue_2443_amv_trace_review.md](issue_2443_amv_trace_review.md), [issue_2259_amv_clipping_success_boundary.md](issue_2259_amv_clipping_success_boundary.md), [issue_2011_amv_actuation_sensitivity_sweep.md](issue_2011_amv_actuation_sensitivity_sweep.md), [issue_2125_seed_sufficiency_ranking_stability.md](issue_2125_seed_sufficiency_ranking_stability.md), [issue_2226_seed_sufficiency_recommendation.md](issue_2226_seed_sufficiency_recommendation.md), [issue_2128_heldout_scenario_family_transfer_protocol.md](issue_2128_heldout_scenario_family_transfer_protocol.md), [issue_2232_planner_mechanism_transfer_benchmark.md](issue_2232_planner_mechanism_transfer_benchmark.md), [issue_2221_static_recenter_transfer.md](issue_2221_static_recenter_transfer.md), [issue_2266_static_recenter_activation.md](issue_2266_static_recenter_activation.md), [issue_2306_static_recenter_activation_trace.md](issue_2306_static_recenter_activation_trace.md), [issue_2402_static_recenter_activation_decision.md](issue_2402_static_recenter_activation_decision.md), [issue_2261_static_recenter_slice_local.md](issue_2261_static_recenter_slice_local.md), [issue_2258_topology_primary_route_audit.md](issue_2258_topology_primary_route_audit.md), [issue_2282_topology_selection_instrumentation.md](issue_2282_topology_selection_instrumentation.md), [issue_2307_topology_score_diagnostic.md](issue_2307_topology_score_diagnostic.md), [issue_3283_amv_actuation_latency_measurement_protocol.md](issue_3283_amv_actuation_latency_measurement_protocol.md), [../code_review.md](../code_review.md) | Fail-closed fallback/degraded handling, artifact classes, simulation-model credibility assessment, rare-event probability language gates and pilot denominator requirements, diagnostic social-compliance and distributional-disruption metric contract boundaries, research-engine scenario-suite proposal boundaries, map/public-smoke result provenance, SocNavBench unavailable/excluded row policy and current control-pipeline asset status, reproducibility, failure-mechanism diagnostics and interpretation taxonomy, mechanism-aware static-deadlock smoke/revise evidence, static-deadlock trace-field reportability evidence, static-deadlock controlled-trace mixed/negative evidence for static-recenter-only and escape-recenter pairs, one-row h500 delayed-rescue evidence, counterfactual pair mechanism-taxonomy/reportability fields, falsification and non-transfer routing, mechanism-aware ranking diagnostics, mechanism-closure status, why-first AMV/topology diagnostic reports and AMV trace-boundary decision, perturbation criticality metric/protocol guidance, predictive-v2 stop/revise decision, AMV actuation evidence ladder, diagnostic ranking result, AMV timeout decomposition, trace-level clipping-vs-progress analysis, field-mapped timeout decision, #2440 closure synthesis, clipping-vs-success boundary, and sensitivity boundaries, seed-sufficiency/ranking-stability diagnostics and recommendations, held-out transfer and planner-mechanism transfer protocol proposals/results, static-recentering activation decision, slice-local evidence, activation-data gap, and narrowed mechanism boundary, topology primary-route audit, selection-score instrumentation, score-overselection diagnostic, the #3293 design-stage evidence-stream integration contract inventory separating calibration/benchmark/operational evidence (non-calibrated, presence-only), AMV actuation-latency and rider-coupling measurement/intake-manifest protocol (blocked-external-input contract; no measured-value claim), and benchmark review traps. |
| Legacy model compatibility | [issue_3469_legacy_ppo_snapshot_parity.md](issue_3469_legacy_ppo_snapshot_parity.md), [../model_registry_publication.md](../model_registry_publication.md), [../../model/registry.md](../../model/registry.md) | Supported legacy PPO checkpoint IDs, unsupported root-local debug snapshots, and the cheap inventory plus opt-in Gymnasium one-step smoke contract. |
| Camera-ready checkpoint provisioning | [issue_4613_camera_ready_checkpoint_provisioning.md](issue_4613_camera_ready_checkpoint_provisioning.md) | Pre-`sbatch` enforced-staged checkpoint preflight gate for camera-ready campaigns, `submit_safe`/`stageable_remote` semantics, persisted `checkpoint_staging.json` report, and the S30 requeue-ahead rules (provisioning-only; no benchmark or paper claim). |
| Issue #3146/#3164 forecast replay fixture suite | [issue_3146_forecast_replay_fixture_suite.md](issue_3146_forecast_replay_fixture_suite.md) | Scenario-diverse frozen-policy diagnostic forecast replay fixture suite, full variant matrix row classifications, tracked #3164 evidence summary, and the negative result that non-`none` variants collapse under shared replay braking. |
| Result-card generator | [issue_3426_result_cards.md](issue_3426_result_cards.md) | Fail-closed bridge from accepted evidence summaries to dissertation-ready Markdown/JSON/LaTeX result-card material without inventing paper-facing claims. |
| External model evidence audits | [issue_2953_qwen_robotnav_metric_audit.md](issue_2953_qwen_robotnav_metric_audit.md) | Source-backed metric crosswalk for Qwen-RobotNav, including the boundary that reported external benchmark numbers are diagnostic context only, not Robot SF benchmark evidence. |
| External data surfaces | [issue_4289_atc_public_surface_terminality.md](issue_4289_atc_public_surface_terminality.md) | ATC public external-data registry/docs/loader terminality, fail-closed absent-data boundary, and out-of-scope private staging or benchmark evidence. |
| SocNavBench / HuNavSim metric correspondence | [issue_2459_socnavbench_hunavsim_mapping.md](issue_2459_socnavbench_hunavsim_mapping.md), [issue_2928_socnavbench_hunavsim_metric_correspondence.md](issue_2928_socnavbench_hunavsim_metric_correspondence.md), [issue_2930_external_benchmark_positioning.md](issue_2930_external_benchmark_positioning.md), [issue_3285_scenario_interop_converter.md](issue_3285_scenario_interop_converter.md), [evidence/issue_3285_closure_audit_2026-07-07.md](evidence/issue_3285_closure_audit_2026-07-07.md) | Interop matrix for safety, efficiency, comfort, personal-space, path-deviation, social-compliance, and human-impact metrics with parity / approximate / unsupported classification; external positioning against CARLA and generic Gymnasium RL environments; dry-run Robot SF → external-benchmark scenario converter emitting a deterministic, schema-validated intermediate representation with explicit unsupported-field reporting (asset-free slice, no cross-benchmark validity claim); closure audit maps merged converter-contract PRs to evidence and keeps runnable external export blocked on staged assets/adapters. |
| Campaign result stores | [issue_3076_campaign_result_store_contract.md](issue_3076_campaign_result_store_contract.md), [issue_3063_campaign_comparison_report.md](issue_3063_campaign_comparison_report.md), [../benchmark.md](../benchmark.md) | Canonical research-engine result-store contract, Parquet/DuckDB fixture proof, required row-status and artifact-provenance fields, analysis-only result-store comparison reports, and the boundary before derived reports or claims. |
| Static recentering diagnostics | [issue_2261_static_recenter_slice_local.md](issue_2261_static_recenter_slice_local.md), [issue_2306_static_recenter_activation_trace.md](issue_2306_static_recenter_activation_trace.md), [issue_2402_static_recenter_activation_decision.md](issue_2402_static_recenter_activation_decision.md), [issue_2438_static_recenter_activation_closure.md](issue_2438_static_recenter_activation_closure.md), [issue_2566_static_recenter_inactive_propagation.md](issue_2566_static_recenter_inactive_propagation.md), [issue_2588_static_deadlock_controlled_trace.md](issue_2588_static_deadlock_controlled_trace.md), [issue_2590_escape_recenter_static_deadlock_controlled_trace.md](issue_2590_escape_recenter_static_deadlock_controlled_trace.md), [issue_2592_static_deadlock_active_row_h500.md](issue_2592_static_deadlock_active_row_h500.md), [research_lane_states.md](research_lane_states.md) | Slice-local boundary, activation trace evidence, requested-field mapping, #2438 current held-out-transfer stop recommendation, #2566 propagation surfaces, #2588/#2590 controlled-trace mixed/negative results, and #2592 h500 delayed-rescue active-row evidence for static recentering. |
| Benchmark release and reports | [benchmark_release_protocol.md](../benchmark_release_protocol.md), [benchmark_camera_ready.md](../benchmark_camera_ready.md), [issue_2040_artifact_publication_workflow.md](issue_2040_artifact_publication_workflow.md), [issue_2037_artifact_compiler_smoke.md](issue_2037_artifact_compiler_smoke.md), [issue_2228_research_dashboard.md](issue_2228_research_dashboard.md), [issue_2571_active_research_queue.md](issue_2571_active_research_queue.md), [issue_2542_dissertation_export_bundle.md](issue_2542_dissertation_export_bundle.md), [issue_2686_release_0_0_2_table_bundle.md](issue_2686_release_0_0_2_table_bundle.md), [issue_2689_release_evidence_handoff_2026_06_15.md](issue_2689_release_evidence_handoff_2026_06_15.md), [issue_3205_release_evidence_snapshot_contract.md](issue_3205_release_evidence_snapshot_contract.md), [issue_3294_release_claim_matrix evidence](evidence/issue_3294_release_claim_matrix/release_claim_matrix.md), [dissertation_claim_export_candidate_report.md](dissertation_claim_export_candidate_report.md), [dissertation_research_bridge.md](dissertation_research_bridge.md), [dissertation_evidence_ledger.md](dissertation_evidence_ledger.md), [dissertation_gap_report.md](dissertation_gap_report.md), [research_lane_states.md](research_lane_states.md), [issue_1542_manuscript_claim_evidence_map.md](issue_1542_manuscript_claim_evidence_map.md), [issue_2153_research_v1_evidence_map.md](issue_2153_research_v1_evidence_map.md), [issue_2943_fast_results_claim_map_v0.md](issue_2943_fast_results_claim_map_v0.md), [issue_2965_release_readiness_dashboard evidence](evidence/issue_2965_release_readiness_dashboard/release_readiness_dashboard.md), [issue_2269_research_v1_trace_case_selection.md](issue_2269_research_v1_trace_case_selection.md), [issue_2280_research_v1_first_trace_review.md](issue_2280_research_v1_first_trace_review.md), [issue_2281_research_v1_trace_review_pack.md](issue_2281_research_v1_trace_review_pack.md), [issue_2309_amv_trace_export_blocker.md](issue_2309_amv_trace_export_blocker.md), [issue_2405_amv_step_export_decision.md](issue_2405_amv_step_export_decision.md), [issue_2263_mechanism_activation_report_fields.md](issue_2263_mechanism_activation_report_fields.md), [issue_2154_ammv_social_force_model.md](issue_2154_ammv_social_force_model.md), [issue_2168_ammv_social_force_pair_diagnostic.md](issue_2168_ammv_social_force_pair_diagnostic.md), [issue_2155_research_v1_ammv_matrix.md](issue_2155_research_v1_ammv_matrix.md), [issue_2172_benchmark_worker_scaling.md](issue_2172_benchmark_worker_scaling.md), [issue_2302_benchmark_worker_scaling.md](issue_2302_benchmark_worker_scaling.md), [issue_2304_benchmark_worker_scaling.md](issue_2304_benchmark_worker_scaling.md), [issue_2214_hot_path_synthesis.md](issue_2214_hot_path_synthesis.md), [issue_3142_fast_pysf_force_optimization.md](issue_3142_fast_pysf_force_optimization.md), [issue_3025_large_crowd_step_profile.md](issue_3025_large_crowd_step_profile.md), [issue_750_paper_results_handoff.md](issue_750_paper_results_handoff.md) | Camera-ready runs, artifact publication workflow, artifact compiler smoke evidence, active research-lane dashboard, Issue #2571 next-cycle research queue, dissertation export bundle pilot/provenance proof, release 0.0.2 table evidence bundle, release evidence handoff snapshot, release evidence snapshot gate, Issue #3294 release claim matrix, dissertation claim-export candidate boundaries, dissertation research bridge, evidence ledger, gap report, and scientific-state table, paper-facing claims, research-v1 AMV claim gates, Issue #2943 fast-results claim map v0 (p0/p1/parked_blocked priority queue aligned with Issue #2910, Issue #2911, Issue #2612, Issue #2937, Issue #2941, and Issue #2923), Issue #2965 release-readiness dashboard evidence, trace-case selection, first trace review, trace-review pack synthesis, AMV trace-export blocker history and single-row step-export proof, mechanism activation report fields, AMMV Social Force model diagnostics, paired AMMV mechanism diagnostics, AMV matrix contract, benchmark worker-scaling and hot-path performance diagnostics, release manifests, step-profile cold/steady-mode semantics, and results handoff. |
| AMMV trace panels | [issue_2428_mechanism_trace_panels.md](issue_2428_mechanism_trace_panels.md), [issue_2430_ammv_trace_annotation.md](issue_2430_ammv_trace_annotation.md), [issue_2432_ammv_trace_selection.md](issue_2432_ammv_trace_selection.md), [issue_2434_ammv_scenario_sweep.md](issue_2434_ammv_scenario_sweep.md), [issue_2227_mechanism_panels.md](issue_2227_mechanism_panels.md), [issue_2405_amv_step_export_decision.md](issue_2405_amv_step_export_decision.md), [evidence/issue_2428_mechanism_trace_panels_2026-06-06/README.md](evidence/issue_2428_mechanism_trace_panels_2026-06-06/README.md), [evidence/issue_2430_ammv_trace_annotation_2026-06-06/README.md](evidence/issue_2430_ammv_trace_annotation_2026-06-06/README.md), [evidence/issue_2432_ammv_trace_selection_2026-06-06/README.md](evidence/issue_2432_ammv_trace_selection_2026-06-06/README.md), [evidence/issue_2434_ammv_scenario_sweep_2026-06-06/README.md](evidence/issue_2434_ammv_scenario_sweep_2026-06-06/README.md) | First diagnostic-only AMMV/default Social Force trace-panel bundle, frame-level parity decision, broader seed-slice selection check, and five-family frame-level/episode-metric screen; the tested adapter-mode AMMV slices remain negative for behavioral-difference evidence, and the #2227 targets need direct mechanism traces, richer AMMV activation instrumentation, or a deliberately more sensitive diagnostic family. |
| Planner integration | [../ai/planner_zoo_context.md](../ai/planner_zoo_context.md), [../benchmark_planner_family_coverage.md](../benchmark_planner_family_coverage.md), [issue_1530_optional_preflight_audit.md](issue_1530_optional_preflight_audit.md), [issue_1360_external_teb_assessment.md](issue_1360_external_teb_assessment.md), [issue_2442_navground_assessment.md](issue_2442_navground_assessment.md), [issue_2550_navground_adapter_spike.md](issue_2550_navground_adapter_spike.md), [issue_2952_qwen_robotnav_assessment.md](issue_2952_qwen_robotnav_assessment.md), [issue_3985_acmpc_feasibility_assessment.md](issue_3985_acmpc_feasibility_assessment.md) | Planner-family coverage, optional planner preflights, adapter provenance, external-framework candidate assessments, Navground adapter spike/design report, Qwen-RobotNav feasibility/asset-tracking decision, ACMPC learned-MPC feasibility scoping, and benchmark readiness. |
| Policy search | [policy_search/INDEX.md](policy_search/INDEX.md), [policy_search/candidate_registry_summary.md](policy_search/candidate_registry_summary.md), [policy_search/candidate_registry.yaml](policy_search/candidate_registry.yaml), [policy_search/contracts/agent_runbook.md](policy_search/contracts/agent_runbook.md), [issue_2453_planner_mechanism_cards.md](issue_2453_planner_mechanism_cards.md) | Current policy-search authorities, planner mechanism cards, candidate lifecycle routing, active execution lanes, and historical/diagnostic boundaries. |
| Topology selection diagnostics | [issue_2258_topology_primary_route_audit.md](issue_2258_topology_primary_route_audit.md), [issue_2307_topology_score_diagnostic.md](issue_2307_topology_score_diagnostic.md), [issue_2393_topology_selection_preflight.md](issue_2393_topology_selection_preflight.md), [issue_2403_topology_selection_score_decision.md](issue_2403_topology_selection_score_decision.md), [issue_2518_topology_near_parity_gate.md](issue_2518_topology_near_parity_gate.md), [issue_2530_topology_near_parity_corrective_smoke.md](issue_2530_topology_near_parity_corrective_smoke.md), [issue_2540_topology_reuse_penalty_diagnostic.md](issue_2540_topology_reuse_penalty_diagnostic.md), [issue_2563_topology_corrective_revision.md](issue_2563_topology_corrective_revision.md), [issue_2570_topology_revise_status_propagation.md](issue_2570_topology_revise_status_propagation.md), [issue_2600_topology_revision_decision.md](issue_2600_topology_revision_decision.md), [issue_2621_topology_revision_hypothesis.md](issue_2621_topology_revision_hypothesis.md), [issue_2624_topology_reuse_penalty_gate.md](issue_2624_topology_reuse_penalty_gate.md), [issue_2704_progress_gated_topology_successor.md](issue_2704_progress_gated_topology_successor.md), [issue_2716_topology_reselection_cross_slice.md](issue_2716_topology_reselection_cross_slice.md), [issue_2742_topology_reselection_successor.md](issue_2742_topology_reselection_successor.md), [issue_2752_topology_reselection_mechanism.md](issue_2752_topology_reselection_mechanism.md), [issue_3463_topology_corrective_behaviors.md](issue_3463_topology_corrective_behaviors.md), [issue_2801_topology_successor_recommendation.md](issue_2801_topology_successor_recommendation.md), [issue_2804_non_topology_successor.md](issue_2804_non_topology_successor.md), [issue_2706_topology_lane_synthesis.md](issue_2706_topology_lane_synthesis.md), [issue_2522_why_first_diagnostics.md](issue_2522_why_first_diagnostics.md) | Primary-route dominance evidence, score-overselection classification, diagnostic-only near-parity selection gate proposal, the Issue #2403 field-mapped `primary_route_overselected` decision, the Issue #2518 accepted diagnostic-only near-parity gate rerun, the Issue #2530 corrective-behavior `revise` result, the Issue #2540 primary-route reuse-penalty diagnostic implementation, the Issue #2563 primary-route reuse-penalty revision proposal, the Issue #2570 status-propagation guard against topology benchmark overclaiming, the Issue #2600 explicit decision to narrow #2540 to the selected #2563 revision, the Issue #2621 post-implementation decision to run the paired reuse-penalty diagnostic gate, the Issue #2624 paired reuse-penalty `revise` result, the Issue #2704 implemented progress-gated successor `revise` result, the Issue #2716 non-canonical cross-slice progress-gated `revise` result, the Issue #2742 clearance-targeted successor launch packet, the Issue #2752 mechanism-level hard-slice failure classification, the Issue #3463 corrective-behavior integration report, the Issue #2801 stop recommendation for topology-reselection-as-clearance on current hard slices, the Issue #2804 non-topology local-policy scoring successor launch packet, the Issue #2706 same-slice selector-variant `stop` synthesis, and the Issue #2522 why-first report preserving the revise boundary. |
| Negative result register | [negative_result_register.md](negative_result_register.md), [evidence/issue_2762_negative_result_register/register.json](evidence/issue_2762_negative_result_register/register.json) | Synthesis/planning aid tracking diagnostic-only, failed, inconclusive, and revise-classified findings so they remain visible in future research planning; not new benchmark or paper evidence. |
| Mechanism-aware evaluation thread | [mechanism_closure_status.md](mechanism_closure_status.md), [issue_2389_mechanism_aware_evaluation_thread.md](issue_2389_mechanism_aware_evaluation_thread.md), [issue_2231_mechanism_aware_ranking.md](issue_2231_mechanism_aware_ranking.md), [issue_2153_research_v1_evidence_map.md](issue_2153_research_v1_evidence_map.md), [issue_2923_mechanism_trace_schema.md](issue_2923_mechanism_trace_schema.md), [issue_2976_mechanism_trace_orca_residuals.md](issue_2976_mechanism_trace_orca_residuals.md), [issue_2981_orca_residual_emission.md](issue_2981_orca_residual_emission.md) | Conservative synthesis scaffold connecting mechanism-closure state to a possible paper/dissertation direction while keeping all current rows diagnostic, blocked, revise, or stop; `mechanism_trace.v1` is the current source-contract schema for local-navigation intervention rows, Issue #2976 adds the diagnostic-only ORCA residual producer fixture, and Issue #2981 adds the scripted ORCA residual emission path from durable fixture input. |
| Social Mini-Game scenario families | [issue_3279_social_mini_game_families.md](issue_3279_social_mini_game_families.md) | Issue #3279 first runnable cut of doorway, hallway, intersection, blind/L-corner, and crowded-traffic local-navigation families as deterministic parameterized generated scenarios with mechanism labels; diagnostic-smoke inputs only (not planner-ranking or benchmark-strength evidence), with width/occlusion/start-timing/yielding-pressure parameter exposure and a full planner smoke deferred. |
| Learned-policy and training | [open_issues_training_split_audit_2026-05-30.md](open_issues_training_split_audit_2026-05-30.md), [policy_search/learned_policy_registry.md](policy_search/learned_policy_registry.md), [policy_search/contracts/learned_local_policy_eligibility.md](policy_search/contracts/learned_local_policy_eligibility.md), [issue_1618_learned_policy_adapter_interface.md](issue_1618_learned_policy_adapter_interface.md), [issue_1966_scenario_belief_interface.md](issue_1966_scenario_belief_interface.md), [issue_2528_scenario_belief_consumer_smoke.md](issue_2528_scenario_belief_consumer_smoke.md), [issue_2565_uncertainty_gating_smoke.md](issue_2565_uncertainty_gating_smoke.md), [issue_2538_scenario_belief_planner_projection.md](issue_2538_scenario_belief_planner_projection.md), [issue_2225_learned_policy_failure_synthesis.md](issue_2225_learned_policy_failure_synthesis.md), [issue_2273_learned_risk_trace_preflight.md](issue_2273_learned_risk_trace_preflight.md), [issue_2312_learned_risk_trace_manifest.md](issue_2312_learned_risk_trace_manifest.md), [issue_2274_hybrid_component_matrix.md](issue_2274_hybrid_component_matrix.md), [issue_2409_local_baseline_quarantine_decision.md](issue_2409_local_baseline_quarantine_decision.md), [issue_2272_orca_residual_launch_packet_status.md](issue_2272_orca_residual_launch_packet_status.md), [issue_2311_orca_residual_lane_decision.md](issue_2311_orca_residual_lane_decision.md), [issue_2271_oracle_imitation_trace_preflight.md](issue_2271_oracle_imitation_trace_preflight.md), [issue_2441_oracle_imitation_trace evidence](evidence/issue_2441_oracle_imitation_traces_2026-06-06/README.md), [issue_2620_oracle_artifact_access.md](issue_2620_oracle_artifact_access.md), [issue_852_queue_fill_batch evidence](evidence/issue_852_queue_fill_batch_2026-06-06/README.md), [issue_2768_learned_prediction_readiness.md](issue_2768_learned_prediction_readiness.md), [issue_2917_scenario_prior_cards.md](issue_2917_scenario_prior_cards.md), [issue_2918_pedestrian_prior_extraction_preflight.md](issue_2918_pedestrian_prior_extraction_preflight.md) | Learned local-policy eligibility, adapter contracts, sensor-agnostic scenario-belief interface design, ScenarioBelief uncertainty-report consumer, stream-gap planner-input gating, and ScenarioBelief-to-planner projection smokes, training blockers, durable checkpoint metadata, local-baseline artifact quarantine closure, negative learned-policy evidence synthesis, learned-risk trace preflight status, ORCA-residual smoke launch status and revise-residual-objective decision, oracle-imitation preflight plus completed-pending-promotion trace collection and blocked artifact-access audit, issue-852 queue-fill diagnostic evidence, the Issue #2768 learned-prediction readiness contract and fail-closed validator, the Issue #2917 ScenarioPrior provenance-card registry, the Issue #2918 external pedestrian-prior extraction staging/preflight contract (blocked-external-input; no calibrated-prior claim), and hybrid-learning component readiness for Issue #1489. |
| Prediction lane execution routing | [prediction_lane_dependency_graph.json](prediction_lane_dependency_graph.json), [issue_2768_learned_prediction_readiness.md](issue_2768_learned_prediction_readiness.md), [issue_2864_forecast_lane_synthesis.md](issue_2864_forecast_lane_synthesis.md), [issue_2929_forecast_workflow_synthesis.md](issue_2929_forecast_workflow_synthesis.md), [paper_closed_loop_forecast_falsification.md](docs/context/paper_closed_loop_forecast_falsification.md), [issue_2902_live_forecast_replay_gate.md](issue_2902_live_forecast_replay_gate.md), [issue_2944_native_cv_closed_loop_smoke.md](issue_2944_native_cv_closed_loop_smoke.md), [issue_2941_native_forecast_replay.md](issue_2941_native_forecast_replay.md), [issue_2960_forecast_planner_consumer.md](issue_2960_forecast_planner_consumer.md), [issue_2966_planner_consumed_forecast_slice.md](issue_2966_planner_consumed_forecast_slice.md), [issue_2937_horizon_denominator_health.md](issue_2937_horizon_denominator_health.md), [issue_3254_predictive_crossing_conflict_negative_result.md](issue_3254_predictive_crossing_conflict_negative_result.md), [issue_3204_proxy_checkpoint_selection_readiness.md](issue_3204_proxy_checkpoint_selection_readiness.md) | Machine-readable dependency graph for forecast/learned-prediction issues, including blockers, dependencies, evidence gates, native CV-only smoke gating, full-variant native replay, post-#2883-to-#2893 synthesis, protocol-frozen closed-loop forecast falsification paper plan, real planner-consumer smoke, full-variant planner-consumed forecast slice, Issue #3254 schema-fixed negative training result, the Issue #3204 proxy-based checkpoint-selection readiness preflight (blocked; diagnostic-only), and preferred execution order before opening blocked lane candidates. Issue #2903 is superseded by #2937 and archived for provenance. |
| CARLA and external simulators | [issue_1169_carla_live_replay.md](issue_1169_carla_live_replay.md), [issue_2122_carla_replay_diagnostics.md](issue_2122_carla_replay_diagnostics.md), [issue_1508_carla_native_aligned_eligibility.md](issue_1508_carla_native_aligned_eligibility.md), [issue_1509_carla_native_fixture_certification.md](issue_1509_carla_native_fixture_certification.md), [issue_2276_carla_parity_lane_decision.md](issue_2276_carla_parity_lane_decision.md), [issue_2014_simulator_backend_matrix.md](issue_2014_simulator_backend_matrix.md), [issue_2016_webots_gazebo_amv_parity_audit.md](issue_2016_webots_gazebo_amv_parity_audit.md), [issue_3028_external_simulator_bridge_preflight.md](issue_3028_external_simulator_bridge_preflight.md), [carla-replay-parity skill](../../.agents/skills/carla-replay-parity/SKILL.md), [issue_1491](https://github.com/ll7/robot_sf_ll7/issues/1491) | CARLA live replay boundaries, diagnostics report boundaries, host-dependent evidence, native/aligned eligibility, fixture certification, parity lane decision routing, simulator backend decision routing, Webots/Gazebo monitor-vs-spike decisions, parity caveats, and Issue #3028 live closed-loop sensor-streaming bridge preflight (scoping note, evidence_tier: idea). |
| Backend adapter contract | [issue_2013_backend_adapter_contract.md](issue_2013_backend_adapter_contract.md), [issue_2014_simulator_backend_matrix.md](issue_2014_simulator_backend_matrix.md), [issue_1646](https://github.com/ll7/robot_sf_ll7/issues/1646), [issue_1491](https://github.com/ll7/robot_sf_ll7/issues/1491) | Backend-agnostic scenario/trace replay contract for alternate simulators, required adapter fields, fail-closed behavior, claim boundaries, and near-term vs monitor-only backend routing. |
| SLURM and long jobs | [slurm_issue_batch_status_2026-05-21.md](slurm_issue_batch_status_2026-05-21.md), [slurm_job_discovery_2026-05-31.md](slurm_job_discovery_2026-05-31.md), [issue_1894_slurm_job_finalizer.md](issue_1894_slurm_job_finalizer.md), [issue_3075_durable_artifact_backend.md](issue_3075_durable_artifact_backend.md), [issue_3425_slurm_to_claim_blocker.md](issue_3425_slurm_to_claim_blocker.md), [../dev/slurm_submission.md](../dev/slurm_submission.md), [../dev/slurm_resource_audit.md](../dev/slurm_resource_audit.md), [open_issues_training_split_audit_2026-05-30.md](open_issues_training_split_audit_2026-05-30.md) | Canonical issue-status ledger, live discovery/dependency routing, launch packets, campaign state, resource limits, artifact finalization, the approved W&B durable artifact backend decision (#3075) plus the finalizer durable-URI contract, Issue #3425 local-machine blocker for the SLURM-to-claim vertical slice, and artifact preservation for long runs. |
| Root layout and cleanup | [root_layout_structured_migration_2026-06-01.md](root_layout_structured_migration_2026-06-01.md), [issue_2035_path_reference_audit.md](issue_2035_path_reference_audit.md), [issue_1573_root_layout_inventory.md](issue_1573_root_layout_inventory.md) | Current root-structure migration, path-reference cleanup validation, and the historical root-layout inventory retained as provenance. Superseded root-layout notes stay discoverable through `catalog.yaml` and the cleanup notes below instead of acting as current entry points. |
| Adversarial search | [issue_4360_adversarial_dispatchable_inventory.md](issue_4360_adversarial_dispatchable_inventory.md), [issue_2468_adversarial_generation_roadmap.md](issue_2468_adversarial_generation_roadmap.md), [issue_3292_rare_event_probability_plan.md](issue_3292_rare_event_probability_plan.md), [issue_3474_seed_overlap_policy.md](issue_3474_seed_overlap_policy.md), [issue_2524_adversarial_manifests.md](issue_2524_adversarial_manifests.md), [issue_2529_llm_manifest_interface.md](issue_2529_llm_manifest_interface.md), [issue_2562_adversarial_manifest_smoke.md](issue_2562_adversarial_manifest_smoke.md), [issue_2567_adversarial_manifest_quality.md](issue_2567_adversarial_manifest_quality.md), [issue_3281_naturalistic_vru_priors.md](issue_3281_naturalistic_vru_priors.md), [issue_2568_adversarial_expansion_gate.md](issue_2568_adversarial_expansion_gate.md), [issue_2618_adversarial_manifest_smoke.md](issue_2618_adversarial_manifest_smoke.md), [issue_2658_adversarial_manifest_smoke.md](issue_2658_adversarial_manifest_smoke.md), [issue_2725_generator_readiness.md](issue_2725_generator_readiness.md), [issue_1457_adversarial_generation_protocol.md](issue_1457_adversarial_generation_protocol.md), [issue_1500_adversarial_manifest.md](issue_1500_adversarial_manifest.md), [issue_1571_adversarial_smoke_packet_sharpening.md](issue_1571_adversarial_smoke_packet_sharpening.md), [issue_1502_adversarial_two_family_run.md](issue_1502_adversarial_two_family_run.md), [issue_1861_adversarial_replay_determinism_gate.md](issue_1861_adversarial_replay_determinism_gate.md), [issue_1878_head_on_route_replay_determinism.md](issue_1878_head_on_route_replay_determinism.md), [issue_1503_adversarial_stress_synthesis.md](issue_1503_adversarial_stress_synthesis.md), [../ai/awesome_copilot_adaptation.md](../ai/awesome_copilot_adaptation.md) | Current #4360 dispatchable inventory and runbook, cross-method adversarial generation roadmap, rare-event probability language gates and compact pilot plan, seed-overlap policy for held-out proposal-vs-random evidence, validator-backed manifest generation, guarded LLM-to-manifest interface, route-materialized planner smoke, compact manifest quality metrics, additive naturalistic VRU prior metadata, learned-expansion gate, generated-manifest collision/low-progress smoke, validator-runner compact evidence smoke, generator-readiness and training-readiness gates, bounded generation, manifest freeze, smoke packets, two-family execution evidence, replay determinism, head-on replay determinism, stress synthesis, and workflow adaptation. |
| Manual control and trace analysis | [issue_1151_manual_control_mvp_foundation.md](issue_1151_manual_control_mvp_foundation.md), [issue_1689_simulation_trace_export_schema.md](issue_1689_simulation_trace_export_schema.md), [issue_2038_real_trace_viewer_smoke.md](issue_2038_real_trace_viewer_smoke.md), [issue_2236_trace_mechanism_evidence_rubric.md](issue_2236_trace_mechanism_evidence_rubric.md), [issue_2463_mechanism_signal_checker.md](issue_2463_mechanism_signal_checker.md), [issue_2543_trace_failure_predicates.md](issue_2543_trace_failure_predicates.md), [issue_2667_trace_failure_predicate_tables.md](issue_2667_trace_failure_predicate_tables.md), [issue_2688_trace_predicate_matrix.md](issue_2688_trace_predicate_matrix.md), [issue_3278_real_trace_validation_contract.md](issue_3278_real_trace_validation_contract.md), [issue_2263_mechanism_activation_report_fields.md](issue_2263_mechanism_activation_report_fields.md), [issue_2227_mechanism_panels.md](issue_2227_mechanism_panels.md), [issue_2428_mechanism_trace_panels.md](issue_2428_mechanism_trace_panels.md), [issue_2270_panel_candidate_manifest.md](issue_2270_panel_candidate_manifest.md), [issue_2405_amv_step_export_decision.md](issue_2405_amv_step_export_decision.md), [issue_2527_waiting_crossing_fixture.md](issue_2527_waiting_crossing_fixture.md), [issue_2564_signal_state_proxy_smoke.md](issue_2564_signal_state_proxy_smoke.md), [issue_2526_cyclist_vru_smoke.md](issue_2526_cyclist_vru_smoke.md), [issue_2223_topology_hypothesis_planning.md](issue_2223_topology_hypothesis_planning.md), [issue_2258_topology_primary_route_audit.md](issue_2258_topology_primary_route_audit.md), [issue_2307_topology_score_diagnostic.md](issue_2307_topology_score_diagnostic.md), [issue_2946_mechanism_figure_pack.md](issue_2946_mechanism_figure_pack.md), [issue_1646_analysis_workbench_closeout.md](issue_1646_analysis_workbench_closeout.md), [../debug_visualization.md](../debug_visualization.md) | Recorder workflows, trace export shape, real-trace viewer smoke evidence, trace-mechanism evidence levels, nonzero mechanism-signal gate, trace-level failure predicates with fail-closed `not_available` rows, denominator-aware predicate-table diagnostic evidence, the proposed predeclared trace-predicate benchmark matrix, the metadata-only real-trace validation-contract checker for candidate micromobility datasets, mechanism activation report fields, mechanism-panel input readiness, first AMMV/default diagnostic trace-panel bundle, first compact mechanism-evidence figure pack from existing tracked traces, candidate-trace blockers, AMMV single-row step-export proof, authored waiting/crossing trace metadata, trace-only signal-state proxy smoke, cyclist-like VRU proxy trace metadata, topology-hypothesis explanation diagnostics, primary-route audit, score-overselection diagnostic, the #1646 analysis-workbench epic closeout audit (criterion→evidence map), and debug visualization boundaries. |

## Catalog Status Cleanup Notes

- [issue_2309_amv_trace_export_blocker.md](issue_2309_amv_trace_export_blocker.md) remains
  linked as historical blocker context; use
  [issue_2405_amv_step_export_decision.md](issue_2405_amv_step_export_decision.md) for the
  current single-row step-export proof.
- [issue_2660_topology_successor_gate.md](archive/issue_2660_topology_successor_gate.md) is superseded
  by [issue_2704_progress_gated_topology_successor.md](issue_2704_progress_gated_topology_successor.md);
  keep Issue #2660 only for the successor-selection decision trail.
- [issue_2903_horizon_denominator_health.md](archive/issue_2903_horizon_denominator_health.md) is
  superseded by [issue_2937_horizon_denominator_health.md](issue_2937_horizon_denominator_health.md)
  and is already struck through in the prediction-lane row above.
- [issue_1690_root_layout_inventory.md](archive/issue_1690_root_layout_inventory.md),
  [issue_1583_high_risk_root_boundaries.md](archive/issue_1583_high_risk_root_boundaries.md), and
  [issue_1598_1599_root_compatibility_decisions.md](archive/issue_1598_1599_root_compatibility_decisions.md)
  are superseded by
  [root_layout_structured_migration_2026-06-01.md](root_layout_structured_migration_2026-06-01.md);
  keep them only as root-layout decision provenance.

Recent static-deadlock h500 evidence: [issue_2594_static_deadlock_broader_h500.md](issue_2594_static_deadlock_broader_h500.md)
records the predeclared broader 3x3 slice after the selected active-row Issue #2592 result. It
supports `broader_delayed_rescue_supported` for the only unsolved active row, not benchmark-candidate
or paper-facing proof.

Recent manifest lineage schema: [issue_2659_lineage_schema_unification.md](issue_2659_lineage_schema_unification.md)
records the shared additive lineage/evidence-boundary fields for `scenario_prior.v1`,
`adversarial_scenario_manifest.v1`, and `counterfactual_scenario_pair.v1`; it is schema
reviewability work only, not benchmark or paper-facing evidence.

Recent signal-state promotion contract: [issue_2662_signal_state_promotion_contract.md](issue_2662_signal_state_promotion_contract.md)
records the fail-closed distinction between `proxy_diagnostic`, `planner_observable`, and
`unavailable` signal-state rows for future signalized-crossing benchmarks; it is schema/trace
contract work only, not benchmark evidence.

Recent signalized-crossing runtime smoke: [evidence/issue_2799_signalized_runtime/README.md](evidence/issue_2799_signalized_runtime/README.md)
records simulator-backed denominator/exclusion evidence for `red_required_stop`, `green_proceed`,
`unavailable_no_claim`, and `proxy_only_denominator_excluded` rows; it proves runtime denominator
plumbing, not traffic-light realism or planner-ranking performance.

Recent GitHub payload audit: [issue_2909_github_payload_overfetch_audit.md](issue_2909_github_payload_overfetch_audit.md)
records the workflow-helper field audit for `scripts/dev/` GitHub calls. Current finding: helper
payloads are mostly required or intentionally opt-in; repeated issue-claim checks are the main
future optimization candidate.

Recent closed-issue state-label cleanup: [issue_3098_closed_state_label_hygiene.md](issue_3098_closed_state_label_hygiene.md)
records the 2026-06-18 REST-backed removal of stale live `state:*` labels from closed issues.

Recent research-engine gap audit: [issue_3058_research_engine_gap_audit.md](issue_3058_research_engine_gap_audit.md)
records live issue/card/artifact mismatches before new empirical campaigns start. It routes
existing conflicts to card correction, label cleanup, follow-up validation policy, or fail-closed
terminal state without upgrading any benchmark or paper-facing claim.

Recent behavior-model variant preflight: [issue_3064_behavior_variants_inventory.md](issue_3064_behavior_variants_inventory.md)
records the fail-closed inventory for native Social Force, AMMV-aware Social Force, and
Social-Navigation-PyEnvs adapter-backed behavior variants. Current status is one
`benchmark_valid_candidate`, one `diagnostic_only`, and four `not_available` rows; diagnostic and
unavailable rows are not benchmark-success evidence.

Recent HSFM + TTC predictive force prototype:
[issue_3481_hsfm_ttc_predictive_forces.md](issue_3481_hsfm_ttc_predictive_forces.md)
records the opt-in `hsfm_ttc_predictive_v1` selector, TTC parameter contract, diagnostic evidence
boundary, and explicit no-benchmark/no-paper-claim scope. The companion body-orientation
alignment-torque slice is in
[issue_3481_hsfm_alignment_torque.md](issue_3481_hsfm_alignment_torque.md): the opt-in
`hsfm_alignment_torque_v1` selector that decouples pedestrian heading `phi` from the instantaneous
force direction via a damped second-order torque (diagnostic/prototype).

Recent heavy forecast-model study/preflight: [forecast_heavy_model_study_2026-06-20.md](forecast_heavy_model_study_2026-06-20.md)
records the analysis-only inventory of heavy predictor families (transformer, AgentFormer-like,
CVAE, diffusion) with literature-derived compute/latency/uncertainty/integration estimates, the
fail-closed offline-evaluation surface probe, and the `blocked` minimum-offline-experiment
status (owner: `robot_sf/research/forecast_heavy_model_inventory.py`, #2845). No model is trained
and no model-quality claim is made.

Recent campaign comparison report: [issue_3063_campaign_comparison_report.md](issue_3063_campaign_comparison_report.md)
records the analysis-only report path from canonical campaign result stores, with tracked fixture
evidence showing row-status caveats, denominators, metric summaries, visual summaries, and
descriptive statistical hooks.

Recent AMV actuation feasibility ranking: [issue_2446_amv_feasibility_ranking.md](issue_2446_amv_feasibility_ranking.md)
records the diagnostic-only conclusion that actuation feasibility is a secondary ranking signal for
the matched AMV actuation-smoke pair, not planner-improvement, benchmark, hardware-calibration, or
paper-facing evidence.

Recent AMV feasibility ranking stress synthesis: [issue_3170_amv_feasibility_ranking_stress.md](issue_3170_amv_feasibility_ranking_stress.md)
records that the available multi-scenario, multi-seed AMMV/default evidence is frame-identical and
the actuation-aware ranking signal remains one-scenario/one-seed only, so no general AMV
feasibility ranking claim is justified yet.

Recent AMV paired actuation feasibility slice: [issue_3181_amv_feasibility_ranking.md](issue_3181_amv_feasibility_ranking.md)
records a 2-scenario x 2-seed synthetic diagnostic comparison where the actuation-aware variant
reduced or tied command clipping, but success stayed zero and no benchmark-strength or paper-facing
ranking claim is supported.

Recent pedestrian-density runtime smoke: [issue_3200_density_runtime_smoke_summary.json](evidence/issue_3200_density_runtime_smoke_summary.json)
records the diagnostic-only same-seed smoke over top coverage-novel and lowest-novelty comparator
rows. All four rows ended `horizon_exhausted`; no benchmark or paper-facing density claim is
promoted.

Recent evidence-catalog backlog: [issue_3014_evidence_catalog_backlog.md](issue_3014_evidence_catalog_backlog.md)
records the 2026-06-19 full evidence-catalog hygiene scan, 116 uncovered tracked evidence bundles
or files, and the bounded split strategy for future Issue #3014 catalog cleanup PRs.

Recent skill consolidation audit: [skill_consolidation_audit_2026-06-20.md](skill_consolidation_audit_2026-06-20.md)
records the Issue #3189 analysis-only inventory of all repo-local skills, overlap clusters,
best-effort usage evidence, and proposed merge/deprecation follow-ups without changing skill
behavior.

Recent map-runner split completion: [issue_3169_map_runner_split_completion.md](issue_3169_map_runner_split_completion.md)
records the current ownership boundary for map-runner episode execution, batch planning,
serial/parallel execution, completed-summary assembly, and temporary private compatibility aliases.

## Context-Pack Manifests

Generated packs are temporary artifacts and should stay under ignored paths such as
`output/context_packs/`. Source-controlled pack definitions live in
[../context_packs/](../context_packs/README.md).

Use [../ai/context_packing.md](../ai/context_packing.md) for the Repomix decision and command
patterns. Start with these curated manifests:

| Pack | Entry point | Use for |
|---|---|---|
| [`learned_policy_integration`](../context_packs/learned_policy_integration.yaml) | `docs/context/policy_search/learned_policy_registry.md` | Learned-policy eligibility, adapter contracts, policy cards, and durable model metadata. |
| [`policy_search`](../context_packs/policy_search.yaml) | `docs/context/policy_search/INDEX.md` | Candidate lifecycle routing, stage-gated execution, promotion gates, and policy-search tooling. |
| [`benchmark_evidence`](../context_packs/benchmark_evidence.yaml) | `docs/context/issue_691_benchmark_fallback_policy.md` | Benchmark fallback policy, review/evidence vocabulary, release surfaces, and compact proof boundaries. |
| [`visualization_workbench`](../context_packs/visualization_workbench.yaml) | `docs/debug_visualization.md` | Diagnostic trace export, analysis-workbench schemas, and benchmark visualization boundaries. |

If an older ad hoc pack scope is still useful, add it under `docs/context_packs/` before relying on
it as a repeatable workflow.

## Optional Tool Boundary

- `context-mode`: optional runtime-layer pilot for long sessions and large tool output. Keep caches
  local and do not make it a repository dependency without a completed evaluation issue.
- `Understand-Anything`: optional read-only graph/navigation evaluation. Pair any graph output with
  this index, [AGENTS.md](../../AGENTS.md), and benchmark fallback policy before drawing workflow
  conclusions.
- Repomix: recommended for static, reproducible context packs. Generated packs are disposable
  `output/` artifacts, not source-of-truth documentation.
