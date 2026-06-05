# Context Retrieval Index

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
| Agent workflow | [goal_driven_agent_loops_2026-05-13.md](goal_driven_agent_loops_2026-05-13.md), [open_issue_execution_improvement_plan_2026-05-30.md](open_issue_execution_improvement_plan_2026-05-30.md), [issue_713_batch_first_issue_workflow.md](issue_713_batch_first_issue_workflow.md), [issue_1776_state_label_routing.md](issue_1776_state_label_routing.md), [Agent Workflow Lessons memory](../../memory/workflows/2026-05-31_agent_workflow_lessons.md) | Issue-to-PR loops, research-result mode, queue exhaustion, batching, issue splitting, state-label routing, live-state checks, delegated-worker proof boundaries, and GitHub workflow policy. |
| Context architecture | This file, [../ai/context_packing.md](../ai/context_packing.md), [../ai/retrieval_deferral.md](../ai/retrieval_deferral.md), [issue_728_coding_agents_compatibility.md](issue_728_coding_agents_compatibility.md) | Context-pack decisions, optional external tools, Markdown-first retrieval, and cross-agent compatibility. |
| Platformization roadmap | [issue_2034_platformization_roadmap.md](issue_2034_platformization_roadmap.md), [issue_2040_artifact_publication_workflow.md](issue_2040_artifact_publication_workflow.md), [issue_2038_real_trace_viewer_smoke.md](issue_2038_real_trace_viewer_smoke.md), [issue_2013_backend_adapter_contract.md](issue_2013_backend_adapter_contract.md), [issue_2015_mujoco_amv_micro_backend.md](issue_2015_mujoco_amv_micro_backend.md), [issue_1894_slurm_job_finalizer.md](issue_1894_slurm_job_finalizer.md), [root_layout_structured_migration_2026-06-01.md](root_layout_structured_migration_2026-06-01.md) | Connected platform layer after June 2026 tooling merges: artifact publication, trace review, backend contracts, optional diagnostic simulator probes, SLURM closeout, external data staging, root-layout cleanup, and next stabilization PRs. |
| Benchmark evidence policy | [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md), [artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md), [issue_1436_reproducibility_flaky_acceptance.md](issue_1436_reproducibility_flaky_acceptance.md), [issue_2012_failure_mechanism_classifier.md](issue_2012_failure_mechanism_classifier.md), [issue_2220_failure_mechanism_taxonomy.md](issue_2220_failure_mechanism_taxonomy.md), [issue_2231_mechanism_aware_ranking.md](issue_2231_mechanism_aware_ranking.md), [issue_2222_perturbation_criticality_metric.md](issue_2222_perturbation_criticality_metric.md), [issue_2234_predictive_perturbation_criticality.md](issue_2234_predictive_perturbation_criticality.md), [issue_2275_predictive_v2_fate.md](issue_2275_predictive_v2_fate.md), [issue_2230_amv_actuation_evidence_ladder.md](issue_2230_amv_actuation_evidence_ladder.md), [issue_2224_amv_actuation_ranking.md](issue_2224_amv_actuation_ranking.md), [issue_2268_amv_timeout_decomposition.md](issue_2268_amv_timeout_decomposition.md), [issue_2011_amv_actuation_sensitivity_sweep.md](issue_2011_amv_actuation_sensitivity_sweep.md), [issue_2125_seed_sufficiency_ranking_stability.md](issue_2125_seed_sufficiency_ranking_stability.md), [issue_2226_seed_sufficiency_recommendation.md](issue_2226_seed_sufficiency_recommendation.md), [issue_2128_heldout_scenario_family_transfer_protocol.md](issue_2128_heldout_scenario_family_transfer_protocol.md), [issue_2232_planner_mechanism_transfer_benchmark.md](issue_2232_planner_mechanism_transfer_benchmark.md), [issue_2221_static_recenter_transfer.md](issue_2221_static_recenter_transfer.md), [issue_2266_static_recenter_activation.md](issue_2266_static_recenter_activation.md), [issue_2258_topology_primary_route_audit.md](issue_2258_topology_primary_route_audit.md), [../code_review.md](../code_review.md) | Fail-closed fallback/degraded handling, artifact classes, reproducibility, failure-mechanism diagnostics and interpretation taxonomy, mechanism-aware ranking diagnostics, perturbation criticality metric/protocol guidance, predictive-v2 stop/revise decision, AMV actuation evidence ladder, diagnostic ranking result, AMV timeout decomposition, and sensitivity boundaries, seed-sufficiency/ranking-stability diagnostics and recommendations, held-out transfer and planner-mechanism transfer protocol proposals/results, static-recentering slice-local evidence and activation-data gap, topology primary-route audit, and benchmark review traps. |
| Benchmark release and reports | [../benchmark_release_protocol.md](../benchmark_release_protocol.md), [../benchmark_camera_ready.md](../benchmark_camera_ready.md), [issue_2040_artifact_publication_workflow.md](issue_2040_artifact_publication_workflow.md), [issue_2037_artifact_compiler_smoke.md](issue_2037_artifact_compiler_smoke.md), [issue_2228_research_dashboard.md](issue_2228_research_dashboard.md), [issue_1542_manuscript_claim_evidence_map.md](issue_1542_manuscript_claim_evidence_map.md), [issue_2153_research_v1_evidence_map.md](issue_2153_research_v1_evidence_map.md), [issue_2154_ammv_social_force_model.md](issue_2154_ammv_social_force_model.md), [issue_2168_ammv_social_force_pair_diagnostic.md](issue_2168_ammv_social_force_pair_diagnostic.md), [issue_2155_research_v1_ammv_matrix.md](issue_2155_research_v1_ammv_matrix.md), [issue_2172_benchmark_worker_scaling.md](issue_2172_benchmark_worker_scaling.md), [issue_2214_hot_path_synthesis.md](issue_2214_hot_path_synthesis.md), [issue_750_paper_results_handoff.md](issue_750_paper_results_handoff.md) | Camera-ready runs, artifact publication workflow, artifact compiler smoke evidence, active research-lane dashboard, paper-facing claims, research-v1 AMV claim gates, AMMV Social Force model diagnostics, paired AMMV mechanism diagnostics, AMV matrix contract, benchmark worker-scaling and hot-path performance diagnostics, release manifests, and results handoff. |
| Planner integration | [../ai/planner_zoo_context.md](../ai/planner_zoo_context.md), [../benchmark_planner_family_coverage.md](../benchmark_planner_family_coverage.md), [issue_1530_optional_preflight_audit.md](issue_1530_optional_preflight_audit.md), [issue_1360_external_teb_assessment.md](issue_1360_external_teb_assessment.md) | Planner-family coverage, optional planner preflights, adapter provenance, and benchmark readiness. |
| Policy search | [policy_search/INDEX.md](policy_search/INDEX.md), [policy_search/candidate_registry_summary.md](policy_search/candidate_registry_summary.md), [policy_search/candidate_registry.yaml](policy_search/candidate_registry.yaml), [policy_search/contracts/agent_runbook.md](policy_search/contracts/agent_runbook.md) | Current policy-search authorities, candidate lifecycle routing, active execution lanes, and historical/diagnostic boundaries. |
| Learned-policy and training | [open_issues_training_split_audit_2026-05-30.md](open_issues_training_split_audit_2026-05-30.md), [policy_search/learned_policy_registry.md](policy_search/learned_policy_registry.md), [policy_search/contracts/learned_local_policy_eligibility.md](policy_search/contracts/learned_local_policy_eligibility.md), [issue_1618_learned_policy_adapter_interface.md](issue_1618_learned_policy_adapter_interface.md), [issue_1966_scenario_belief_interface.md](issue_1966_scenario_belief_interface.md), [issue_2225_learned_policy_failure_synthesis.md](issue_2225_learned_policy_failure_synthesis.md), [issue_2274_hybrid_component_matrix.md](issue_2274_hybrid_component_matrix.md) | Learned local-policy eligibility, adapter contracts, sensor-agnostic scenario-belief interface design, training blockers, durable checkpoint metadata, negative learned-policy evidence synthesis, and hybrid-learning component readiness for Issue #1489. |
| CARLA and external simulators | [issue_1169_carla_live_replay.md](issue_1169_carla_live_replay.md), [issue_2122_carla_replay_diagnostics.md](issue_2122_carla_replay_diagnostics.md), [issue_1508_carla_native_aligned_eligibility.md](issue_1508_carla_native_aligned_eligibility.md), [issue_1509_carla_native_fixture_certification.md](issue_1509_carla_native_fixture_certification.md), [issue_2014_simulator_backend_matrix.md](issue_2014_simulator_backend_matrix.md), [issue_2016_webots_gazebo_amv_parity_audit.md](issue_2016_webots_gazebo_amv_parity_audit.md), [carla-replay-parity skill](../../.agents/skills/carla-replay-parity/SKILL.md), [issue_1491](https://github.com/ll7/robot_sf_ll7/issues/1491) | CARLA live replay boundaries, diagnostics report boundaries, host-dependent evidence, native/aligned eligibility, fixture certification, simulator backend decision routing, Webots/Gazebo monitor-vs-spike decisions, and parity caveats. |
| Backend adapter contract | [issue_2013_backend_adapter_contract.md](issue_2013_backend_adapter_contract.md), [issue_2014_simulator_backend_matrix.md](issue_2014_simulator_backend_matrix.md), [issue_1646](https://github.com/ll7/robot_sf_ll7/issues/1646), [issue_1491](https://github.com/ll7/robot_sf_ll7/issues/1491) | Backend-agnostic scenario/trace replay contract for alternate simulators, required adapter fields, fail-closed behavior, claim boundaries, and near-term vs monitor-only backend routing. |
| SLURM and long jobs | [slurm_issue_batch_status_2026-05-21.md](slurm_issue_batch_status_2026-05-21.md), [slurm_job_discovery_2026-05-31.md](slurm_job_discovery_2026-05-31.md), [issue_1894_slurm_job_finalizer.md](issue_1894_slurm_job_finalizer.md), [../dev/slurm_submission.md](../dev/slurm_submission.md), [../dev/slurm_resource_audit.md](../dev/slurm_resource_audit.md), [open_issues_training_split_audit_2026-05-30.md](open_issues_training_split_audit_2026-05-30.md) | Canonical issue-status ledger, live discovery/dependency routing, launch packets, campaign state, resource limits, artifact finalization, and artifact preservation for long runs. |
| Root layout and cleanup | [root_layout_structured_migration_2026-06-01.md](root_layout_structured_migration_2026-06-01.md), [issue_2035_path_reference_audit.md](issue_2035_path_reference_audit.md), [issue_1690_root_layout_inventory.md](issue_1690_root_layout_inventory.md), [issue_1573_root_layout_inventory.md](issue_1573_root_layout_inventory.md), [issue_1583_high_risk_root_boundaries.md](issue_1583_high_risk_root_boundaries.md), [issue_1598_1599_root_compatibility_decisions.md](issue_1598_1599_root_compatibility_decisions.md) | Current root-structure migration, path-reference cleanup validation, historical root hygiene inventories, high-risk path boundaries, and compatibility shims. |
| Adversarial search | [issue_1457_adversarial_generation_protocol.md](issue_1457_adversarial_generation_protocol.md), [issue_1500_adversarial_manifest.md](issue_1500_adversarial_manifest.md), [issue_1571_adversarial_smoke_packet_sharpening.md](issue_1571_adversarial_smoke_packet_sharpening.md), [issue_1502_adversarial_two_family_run.md](issue_1502_adversarial_two_family_run.md), [issue_1861_adversarial_replay_determinism_gate.md](issue_1861_adversarial_replay_determinism_gate.md), [issue_1878_head_on_route_replay_determinism.md](issue_1878_head_on_route_replay_determinism.md), [issue_1503_adversarial_stress_synthesis.md](issue_1503_adversarial_stress_synthesis.md), [../ai/awesome_copilot_adaptation.md](../ai/awesome_copilot_adaptation.md) | Bounded adversarial generation, manifest freeze, smoke packets, two-family execution evidence, replay determinism, head-on replay determinism, stress synthesis, and workflow adaptation. |
| Manual control and trace analysis | [issue_1151_manual_control_mvp_foundation.md](issue_1151_manual_control_mvp_foundation.md), [issue_1689_simulation_trace_export_schema.md](issue_1689_simulation_trace_export_schema.md), [issue_2038_real_trace_viewer_smoke.md](issue_2038_real_trace_viewer_smoke.md), [issue_2236_trace_mechanism_evidence_rubric.md](issue_2236_trace_mechanism_evidence_rubric.md), [issue_2227_mechanism_panels.md](issue_2227_mechanism_panels.md), [issue_2223_topology_hypothesis_planning.md](issue_2223_topology_hypothesis_planning.md), [issue_2258_topology_primary_route_audit.md](issue_2258_topology_primary_route_audit.md), [../debug_visualization.md](../debug_visualization.md) | Recorder workflows, trace export shape, real-trace viewer smoke evidence, trace-mechanism evidence levels, mechanism-panel input readiness, topology-hypothesis explanation diagnostics, primary-route audit, and debug visualization boundaries. |

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
