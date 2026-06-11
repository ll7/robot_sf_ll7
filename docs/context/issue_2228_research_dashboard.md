# Issue #2228 Research Dashboard

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2228>

Status: current navigation dashboard as of 2026-06-07.

## Purpose

This dashboard keeps active Robot SF research lanes in one place so agents and maintainers can
choose the next research-result issue without re-reading every synthesis note. It is an index and
decision surface, not a new benchmark result. The source notes linked below remain authoritative
for exact commands, metrics, artifacts, and claim boundaries.

Status vocabulary:

- `paper_candidate`: evidence may support manuscript-facing language after the linked proof surface
  is checked for scope, provenance, and caveats.
- `candidate`: evidence is useful for research direction, but needs a stronger or broader proof
  before paper-facing use.
- `diagnostic`: executable or trace evidence exists, but it is local, mechanism-only, adapter-only,
  or otherwise not benchmark-strength.
- `proposal`: a protocol or issue contract exists, but no qualifying execution has run.
- `blocked`: the next proof depends on missing runtime support, provenance, or unavailable inputs.
- `negative`: the tested direction did not support promotion under the recorded contract.

## Active Research Lanes

| Lane | Current support | Evidence tier | Main blocker or risk | Next decision | Canonical sources |
| --- | --- | --- | --- | --- | --- |
| Manuscript claim map and camera-ready boundaries | Several areas are explicitly mapped, but most claims remain blocked, negative, insufficient, or diagnostic-only. | `candidate` for the map; mixed per claim | Treating the map as a result instead of a conservative gate would overstate current support. | Use the map to decide which claim, if any, can move after a linked proof run; keep unsupported claims out of paper-facing language. | [issue_1542_manuscript_claim_evidence_map.md](issue_1542_manuscript_claim_evidence_map.md) |
| Research-v1 AMV matrix and scenario criticality | AMV matrix and selected scenario-criticality claims are candidate directions; AMV model behavior and transfer remain diagnostic; calibrated actuation is blocked. | `candidate`/`diagnostic`/`blocked` by claim ID | The benchmark matrix is not yet a complete, provenance-backed result, and AMV actuation fields remain synthetic or unavailable. | Either execute the named matrix with durable provenance or narrow the matrix to mechanism-only AMMV diagnostics. | [issue_2153_research_v1_evidence_map.md](issue_2153_research_v1_evidence_map.md), [issue_2155_research_v1_ammv_matrix.md](issue_2155_research_v1_ammv_matrix.md) |
| AMMV-aware Social Force and AMV actuation validity | Direct `SocialForcePlanner` mechanism probe activates the AMMV term, adapter benchmark rows are identical and lack AMMV metadata, and the #2249/#2224 matched smoke row shows actuation-aware scoring reduced command clipping from `0.2750` to `0.1875` while success stayed `0.0000` for both candidates. The #2522 why-first report preserves this as route-progress blocked after feasibility improvement. | `diagnostic`/`negative` for success improvement | Adapter-mode rows cannot prove AMMV-vs-default benchmark behavior; calibrated pedestrian lateral deviation and speed adaptation remain unsupported; the actuation-aware row is one episode, synthetic-only, and both compared candidates timed out with `timeout_low_progress`. | Separate feasibility instrumentation from navigation success. Use issue [#2259](https://github.com/ll7/robot_sf_ll7/issues/2259) to explain the timeout driver before proposing another actuation scorer or broader matched slice. | [issue_2168_ammv_social_force_pair_diagnostic.md](issue_2168_ammv_social_force_pair_diagnostic.md), [issue_2154_ammv_social_force_model.md](issue_2154_ammv_social_force_model.md), [issue_2224_amv_actuation_ranking.md](issue_2224_amv_actuation_ranking.md), [issue_2522_why_first_diagnostics.md](issue_2522_why_first_diagnostics.md), [issue_2001_amv_actuation_proxy_source_analysis.md](issue_2001_amv_actuation_proxy_source_analysis.md), [issue_2011_amv_actuation_sensitivity_sweep.md](issue_2011_amv_actuation_sensitivity_sweep.md), [issue #2259](https://github.com/ll7/robot_sf_ll7/issues/2259) |
| Hybrid component effects and static recentering | Static recentering and recenter-after-escape are supported on the local h500 one-factor slice, but the #2250/#2221 held-out family smoke showed no terminal-metric transfer lift and the #2438 closure classified the unsolved held-out row as `mechanism_inactive`. | `diagnostic` for local h500 design; `negative`/`inactive` for the current held-out transfer route | The discovery effect is local diagnostic evidence. The current held-out route should not be rerun or promoted because activation count was zero, command source did not change, and trajectory delta was `0.0 m`. | Stop the current held-out transfer route. Revisit static recentering only through a predeclared activation-targeted static-obstacle slice that preserves activation and command-source fields. | [issue_2182_component_effect_synthesis.md](issue_2182_component_effect_synthesis.md), [issue_2180_one_factor_h500.md](issue_2180_one_factor_h500.md), [issue_2221_static_recenter_transfer.md](issue_2221_static_recenter_transfer.md), [issue_2438_static_recenter_activation_closure.md](issue_2438_static_recenter_activation_closure.md), [issue_2566_static_recenter_inactive_propagation.md](issue_2566_static_recenter_inactive_propagation.md) |
| Planner mechanism transfer | Protocol exists for separating slice-local mechanisms from transferable mechanisms; static recentering is stopped on the current held-out transfer route after #2438, and the topology near-parity lane is `revise` after #2530. | `diagnostic`/`negative` for mitigation | Static recentering is inactive on the inspected held-out row, and topology has not passed a corrective-behavior gate. Neither lane supports transfer or mitigation promotion. | Keep topology transfer or benchmark claims frozen until #2540 or a narrower #2563-derived diagnostic shows progress beyond the single canonical slice; keep static recentering frozen unless a new activation-targeted slice is predeclared. | [issue_2232_planner_mechanism_transfer_benchmark.md](issue_2232_planner_mechanism_transfer_benchmark.md), [issue_2221_static_recenter_transfer.md](issue_2221_static_recenter_transfer.md), [issue_2438_static_recenter_activation_closure.md](issue_2438_static_recenter_activation_closure.md), [issue_2530_topology_near_parity_corrective_smoke.md](issue_2530_topology_near_parity_corrective_smoke.md), [issue_2563_topology_corrective_revision.md](issue_2563_topology_corrective_revision.md), [issue #2540](https://github.com/ll7/robot_sf_ll7/issues/2540) |
| Perturbation criticality and phase sensitivity | Cross-trap and intersection-wait diagnostics now have a reusable `criticality_metric.v1` classification; high-signal families are speed/start-delay in `francis2023_intersection_wait`, while robot-route, wait-duration, and density lanes are low-sensitivity or smoke-only; predictive validity remains untested. | `diagnostic` | Current evidence is local and terminal outcomes often stay unchanged, so it supports mechanism investigation rather than robustness or causal claims. The metric is defined, but not yet proven predictive. | Stop adding new perturbation families until either the #2234 held-out protocol tests predictive value on the high-signal family or a writer emits `criticality_metric.v1` rows from paired outputs. | [issue_2222_perturbation_criticality_metric.md](issue_2222_perturbation_criticality_metric.md), [issue_1965_perturbation_criticality_synthesis.md](issue_1965_perturbation_criticality_synthesis.md), [issue_2234_predictive_perturbation_criticality.md](issue_2234_predictive_perturbation_criticality.md), [issue_1933_perturbation_seed_coverage.md](issue_1933_perturbation_seed_coverage.md), [issue_1937_ped_route_offset.md](issue_1937_ped_route_offset.md), [issue_1941_ped_timing_phase.md](issue_1941_ped_timing_phase.md), [issue_1943_ped_speed_perturbation.md](issue_1943_ped_speed_perturbation.md), [issue_2236_trace_mechanism_evidence_rubric.md](issue_2236_trace_mechanism_evidence_rubric.md) |
| Failure mechanism taxonomy | A shared vocabulary now classifies observed mechanisms, supported hypotheses, weak hypotheses, and unknowns across recent diagnostics. | `candidate` for review language; diagnostic as evidence | The taxonomy is not yet wired into benchmark reports or automated trace review; weak hypotheses should not become conclusions. | Use the vocabulary in new result PRs; next useful proof is a classifier/reporting crosswalk or focused trace review for the strongest seed partitions. | [issue_2220_failure_mechanism_taxonomy.md](issue_2220_failure_mechanism_taxonomy.md), [issue_2012_failure_mechanism_classifier.md](issue_2012_failure_mechanism_classifier.md) |
| Trace-based explanation | Rubric defines four evidence levels from qualitative illustration through cross-case mechanism evidence and maps recent traces to those levels. | `candidate` for review discipline | Trace artifacts can illustrate mechanisms, but single-case or aggregate-only traces do not establish cross-case causality. | Apply the rubric in research-result PRs; defer template/tooling changes until repeated PRs show the checklist is needed. | [issue_2236_trace_mechanism_evidence_rubric.md](issue_2236_trace_mechanism_evidence_rubric.md), [issue_2038_real_trace_viewer_smoke.md](issue_2038_real_trace_viewer_smoke.md) |
| Topology limitation and recovery | Topology-hypothesis probes can expose alternate bottleneck routes, and near-parity gating can produce non-primary selections, but #2530 still classified corrective behavior as `revise`. #2563 now selects a primary-route reuse penalty as the next diagnostic hypothesis, and #2522 preserves the why-first classification as `topology_signal_without_route_progress`. | `diagnostic`/`revise` for mitigation | Selection diversity and non-primary command influence are not enough: the canonical slice still ended `horizon_exhausted`, so topology remains diagnostic-only. | Test the #2563 reuse-penalty hypothesis in #2540 or a narrower child before any topology benchmark, leaderboard, or transfer claim. | [issue_2258_topology_primary_route_audit.md](issue_2258_topology_primary_route_audit.md), [issue_2403_topology_selection_score_decision.md](issue_2403_topology_selection_score_decision.md), [issue_2518_topology_near_parity_gate.md](issue_2518_topology_near_parity_gate.md), [issue_2530_topology_near_parity_corrective_smoke.md](issue_2530_topology_near_parity_corrective_smoke.md), [issue_2563_topology_corrective_revision.md](issue_2563_topology_corrective_revision.md), [issue_2522_why_first_diagnostics.md](issue_2522_why_first_diagnostics.md), [issue #2540](https://github.com/ll7/robot_sf_ll7/issues/2540) |
| Learned-policy failures and next interfaces | Recent synthesis supports stopping repeated generic BC warm-start PPO continuations and prioritizing residual-over-ORCA or learned-risk-surface interfaces. | `diagnostic`/`negative` for repeated generic lanes | The conclusion is about tested failure modes, not a global claim that learned methods are ineffective. | Require a nominal sanity gate, explicit checkpoint/data provenance, and mechanism-level diagnostics before new learned-policy benchmark claims. | [issue_2225_learned_policy_failure_synthesis.md](issue_2225_learned_policy_failure_synthesis.md), [policy_search/learned_policy_registry.md](policy_search/learned_policy_registry.md), [policy_search/contracts/learned_local_policy_eligibility.md](policy_search/contracts/learned_local_policy_eligibility.md) |
| Seed sufficiency and ranking stability | Analyzer exists for seed-level report bundles and can expose interval width, rank flips, scenario-family instability, and advisory single-seed surfaces. | `diagnostic` | Analyzer output diagnoses uncertainty; it does not create scenario coverage, significance, or paper-ready planner superiority. | Use issue [#2226](https://github.com/ll7/robot_sf_ll7/issues/2226) or a named campaign bundle to decide whether a larger pre-specified seed schedule is needed. Attach confidence bands and S5/S10/S20 gate status to any ranking claim. | [issue_2125_seed_sufficiency_ranking_stability.md](issue_2125_seed_sufficiency_ranking_stability.md), [issue_832_paper_matrix_extended_seed_schedule.md](issue_832_paper_matrix_extended_seed_schedule.md), [issue_1608_seed_sensitivity_analysis.md](issue_1608_seed_sensitivity_analysis.md), [issue_595_seed_variability_contract.md](issue_595_seed_variability_contract.md), [issue_1545_power_aware_seed_budget_planning.md](issue_1545_power_aware_seed_budget_planning.md) |

## Queue Guidance

Issue #2571 adds a compact next-cycle queue for the current research focus: adversarial manifest
quality, oracle trace artifact access, and coupled signal-state/ScenarioBelief uncertainty
semantics. Use [issue_2571_active_research_queue.md](issue_2571_active_research_queue.md) as the
handoff surface for that queue, while keeping GitHub issues as the executable work system.

Given the current evidence state, the next research-result issues should favor work that moves a
claim boundary rather than adding another local diagnostic with the same limitation:

1. Do not repeat one-seed static-recenter or topology-hypothesis diagnostics unless the mechanism
   changes; #2250/#2221 is `slice_local` and #2251/#2223 is explanation-only without mitigation.
2. Run the #2234 held-out perturbation protocol only after the #2222 `criticality_metric.v1`
   contract is the accepted schema for paired outputs, and keep the current perturbation lane at
   `predictive validity untested` until that proof exists.
3. Explain the #2249/#2224 feasibility-versus-success split before improving AMMV/AMV benchmark
   metadata or running a broader matched AMV actuation slice; issue #2259 is the next empirical
   follow-through.
4. Keep topology benchmark and leaderboard claims frozen until the #2563 primary-route
   reuse-penalty hypothesis is tested in a paired diagnostic such as issue #2540.
5. Keep adversarial RL/diffusion expansion gated on
   [issue_2568_adversarial_expansion_gate.md](issue_2568_adversarial_expansion_gate.md): #2562
   smoke and #2567 quality metrics are prerequisite diagnostics, not benchmark evidence. Keep
   oracle-imitation training, including Issue #1496 and Issue #2569, gated on
   [Issue #2620](issue_2620_oracle_artifact_access.md) until durable trace artifact access exists; the Issue #2620 audit preserves
   `artifact_retrieval_blocked` because the Issue #2441 raw traces have no durable pointer and are
   absent from the current machine.
6. Use learned-policy work only when it changes the interface or proof gate, not when it repeats the
   generic continuation path already classified in the learned-policy synthesis.
7. Use simulator-speed work when the research queue is exhausted or when speed unlocks a named
   benchmark/proof run.

## Update Triggers

Refresh this dashboard when any of the following happen:

- a linked issue changes a lane from `proposal` or `diagnostic` to a stronger evidence tier;
- a benchmark report changes paper-facing claim support, fallback/degraded status, or seed
  sufficiency;
- a new synthesis closes a research parent issue or revises the next decision;
- a lane becomes blocked by unavailable runtime support or missing durable provenance.

## Validation

This issue #2260/#2228 update was validated as a docs-only navigation change:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
