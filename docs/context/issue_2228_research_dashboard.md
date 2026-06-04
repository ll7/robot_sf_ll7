# Issue #2228 Research Dashboard

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2228>

Status: current navigation dashboard as of 2026-06-04.

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
| AMMV-aware Social Force and AMV actuation validity | Direct `SocialForcePlanner` mechanism probe activates the AMMV term, but adapter benchmark rows are identical and lack AMMV metadata. | `diagnostic` | Adapter-mode rows cannot prove AMMV-vs-default benchmark behavior; calibrated pedestrian lateral deviation and speed adaptation remain unsupported. | Choose between wiring AMMV diagnostics into an executable benchmark path or keeping AMMV as a mechanism-only variant. | [issue_2168_ammv_social_force_pair_diagnostic.md](issue_2168_ammv_social_force_pair_diagnostic.md), [issue_2154_ammv_social_force_model.md](issue_2154_ammv_social_force_model.md), [issue_2001_amv_actuation_proxy_source_analysis.md](issue_2001_amv_actuation_proxy_source_analysis.md), [issue_2011_amv_actuation_sensitivity_sweep.md](issue_2011_amv_actuation_sensitivity_sweep.md) |
| Hybrid component effects and static recentering | Static recentering and recenter-after-escape are supported on the local h500 one-factor slice; static escape alone, corridor-transit terms, selector-only, and speed/progress 2.4 are lower priority. | `candidate` for planner design; not paper-facing causality | The effect is from an 18-row local diagnostic slice, so one-row and two-row deltas cannot establish general causality. | Test static recentering under the planner-mechanism transfer protocol before promoting it beyond the discovery slice. | [issue_2182_component_effect_synthesis.md](issue_2182_component_effect_synthesis.md), [issue_2180_one_factor_h500.md](issue_2180_one_factor_h500.md), [issue_2232_planner_mechanism_transfer_benchmark.md](issue_2232_planner_mechanism_transfer_benchmark.md) |
| Planner mechanism transfer | Protocol exists for separating slice-local mechanisms from transferable mechanisms; static recentering is the first recommended candidate. | `proposal` | No follow-up transfer run has executed the protocol; fallback, degraded, failed, or unavailable rows cannot count as support. | Run issue [#2221](https://github.com/ll7/robot_sf_ll7/issues/2221) for static recentering, then classify as `transfer_supported`, `slice_local`, `trade_off`, or `inconclusive`. | [issue_2232_planner_mechanism_transfer_benchmark.md](issue_2232_planner_mechanism_transfer_benchmark.md), [issue_2128_heldout_scenario_family_transfer_protocol.md](issue_2128_heldout_scenario_family_transfer_protocol.md) |
| Perturbation criticality and phase sensitivity | Cross-trap and intersection-wait diagnostics show useful local perturbation signals, especially phase/speed sensitivity in `francis2023_intersection_wait`; several low-magnitude route or wait perturbations were low-sensitivity. | `diagnostic` | Current evidence is local and terminal outcomes often stay unchanged, so it supports mechanism investigation rather than robustness claims. | Prefer schema-backed perturbation summaries and trace-backed criticality explanations before broadening; issue [#2234](https://github.com/ll7/robot_sf_ll7/issues/2234) is the next compact synthesis lane. Promote one perturbation family at a time with explicit pass/fail thresholds. | [issue_1965_perturbation_criticality_synthesis.md](issue_1965_perturbation_criticality_synthesis.md), [issue_1933_perturbation_seed_coverage.md](issue_1933_perturbation_seed_coverage.md), [issue_1937_ped_route_offset.md](issue_1937_ped_route_offset.md), [issue_1939_corridor_trace_response.md](issue_1939_corridor_trace_response.md), [issue_1941_ped_timing_phase.md](issue_1941_ped_timing_phase.md), [issue_2236_trace_mechanism_evidence_rubric.md](issue_2236_trace_mechanism_evidence_rubric.md) |
| Failure mechanism taxonomy | A shared vocabulary now classifies observed mechanisms, supported hypotheses, weak hypotheses, and unknowns across recent diagnostics. | `candidate` for review language; diagnostic as evidence | The taxonomy is not yet wired into benchmark reports or automated trace review; weak hypotheses should not become conclusions. | Use the vocabulary in new result PRs; next useful proof is a classifier/reporting crosswalk or focused trace review for the strongest seed partitions. | [issue_2220_failure_mechanism_taxonomy.md](issue_2220_failure_mechanism_taxonomy.md), [issue_2012_failure_mechanism_classifier.md](issue_2012_failure_mechanism_classifier.md) |
| Trace-based explanation | Rubric defines four evidence levels from qualitative illustration through cross-case mechanism evidence and maps recent traces to those levels. | `candidate` for review discipline | Trace artifacts can illustrate mechanisms, but single-case or aggregate-only traces do not establish cross-case causality. | Apply the rubric in research-result PRs; defer template/tooling changes until repeated PRs show the checklist is needed. | [issue_2236_trace_mechanism_evidence_rubric.md](issue_2236_trace_mechanism_evidence_rubric.md), [issue_2038_real_trace_viewer_smoke.md](issue_2038_real_trace_viewer_smoke.md) |
| Topology limitation and recovery | Topology-hypothesis probes can expose alternate bottleneck routes and command-source dominance on a named slice; older recovery notes show partial corridor/topology recovery but not broad closure. | `diagnostic` | The probe is not a planner implementation and should not be cited as benchmark success evidence. Partial recovery can hide family-specific failures. | Use topology traces to scope a bounded recovery planner or mechanism-transfer test, then require native execution, family-level inclusion/exclusion rules, and row-status accounting. | [issue_1692_topology_hypothesis_probe.md](issue_1692_topology_hypothesis_probe.md), [issue_1028_corridor_subgoal_recovery.md](issue_1028_corridor_subgoal_recovery.md), [issue_1034_continuous_corridor_maneuver.md](issue_1034_continuous_corridor_maneuver.md), [issue_2220_failure_mechanism_taxonomy.md](issue_2220_failure_mechanism_taxonomy.md) |
| Learned-policy failures and next interfaces | Recent synthesis supports stopping repeated generic BC warm-start PPO continuations and prioritizing residual-over-ORCA or learned-risk-surface interfaces. | `diagnostic`/`negative` for repeated generic lanes | The conclusion is about tested failure modes, not a global claim that learned methods are ineffective. | Require a nominal sanity gate, explicit checkpoint/data provenance, and mechanism-level diagnostics before new learned-policy benchmark claims. | [issue_2225_learned_policy_failure_synthesis.md](issue_2225_learned_policy_failure_synthesis.md), [policy_search/learned_policy_registry.md](policy_search/learned_policy_registry.md), [policy_search/contracts/learned_local_policy_eligibility.md](policy_search/contracts/learned_local_policy_eligibility.md) |
| Seed sufficiency and ranking stability | Analyzer exists for seed-level report bundles and can expose interval width, rank flips, scenario-family instability, and advisory single-seed surfaces. | `diagnostic` | Analyzer output diagnoses uncertainty; it does not create scenario coverage, significance, or paper-ready planner superiority. | Use issue [#2226](https://github.com/ll7/robot_sf_ll7/issues/2226) or a named campaign bundle to decide whether a larger pre-specified seed schedule is needed. Attach confidence bands and S5/S10/S20 gate status to any ranking claim. | [issue_2125_seed_sufficiency_ranking_stability.md](issue_2125_seed_sufficiency_ranking_stability.md), [issue_832_paper_matrix_extended_seed_schedule.md](issue_832_paper_matrix_extended_seed_schedule.md), [issue_1608_seed_sensitivity_analysis.md](issue_1608_seed_sensitivity_analysis.md), [issue_595_seed_variability_contract.md](issue_595_seed_variability_contract.md), [issue_1545_power_aware_seed_budget_planning.md](issue_1545_power_aware_seed_budget_planning.md) |

## Queue Guidance

Given the current evidence state, the next research-result issues should favor work that moves a
claim boundary rather than adding another local diagnostic with the same limitation:

1. Run or tighten mechanism-transfer evidence for static recentering if issue #2221 is ready.
2. Synthesize perturbation criticality into a schema-backed decision surface if issue #2234 remains
   the smallest available research result.
3. Improve AMMV/AMV benchmark metadata only if the work can expose actuation-specific evidence in an
   executable path; otherwise keep AMMV mechanism-only.
4. Use learned-policy work only when it changes the interface or proof gate, not when it repeats the
   generic continuation path already classified in the learned-policy synthesis.
5. Use simulator-speed work when the research queue is exhausted or when speed unlocks a named
   benchmark/proof run.

## Update Triggers

Refresh this dashboard when any of the following happen:

- a linked issue changes a lane from `proposal` or `diagnostic` to a stronger evidence tier;
- a benchmark report changes paper-facing claim support, fallback/degraded status, or seed
  sufficiency;
- a new synthesis closes a research parent issue or revises the next decision;
- a lane becomes blocked by unavailable runtime support or missing durable provenance.

## Validation

This issue #2228 update was validated as a docs-only navigation change:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
