# Dissertation Claim-Export Candidate Report

Issue: [#2552](https://github.com/ll7/robot_sf_ll7/issues/2552)
Status: conservative synthesis/reporting aid; not dissertation, benchmark, safety, or paper-grade
evidence.

## Purpose

This report identifies current Robot SF evidence surfaces that are safe candidate sources for
dissertation planning language. It is a claim boundary map, not manuscript prose. Rows marked
`diagnostic`, `blocked`, or `future-work only` must not be used as Results claims unless a later
proof surface upgrades them with durable execution evidence, provenance, metrics, and limitations.

Readiness vocabulary:

- `release-backed`: tracked benchmark/report evidence may support carefully scoped Results wording
  after the cited source is checked for caveats.
- `diagnostic`: useful for Discussion, methods, or failure-analysis framing only.
- `blocked`: a named evidence or provenance blocker prevents manuscript claim use.
- `future-work only`: protocol, roadmap, or open issue direction without qualifying evidence.

## Candidate Rows

| Claim candidate | Evidence source | Artifact category | Allowed wording | Not-claimed boundary | Chapter fit | Readiness |
| --- | --- | --- | --- | --- | --- | --- |
| Robot SF has a reproducible AMV benchmark/evaluation scaffold with explicit planner rows, scenario/config surfaces, and fail-closed caveats. | [issue_1542_manuscript_claim_evidence_map.md](issue_1542_manuscript_claim_evidence_map.md), [issue_1344_paired_amv_protocol_report.md](issue_1344_paired_amv_protocol_report.md), [issue_1353_broader_amv_preflight.md](issue_1353_broader_amv_preflight.md), [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md) | Benchmark/report synthesis | "The repository defines and exercises AMV evaluation surfaces with explicit fallback/unavailable caveats." | Do not claim full AMV realism, calibrated actuation, human-centered validity, or cross-simulator transfer from these rows. | Results, with caveats; Methods for protocol framing | `release-backed` |
| Research-v1 AMV matrix and scenario criticality are organized enough to guide future campaigns. | [issue_2153_research_v1_evidence_map.md](issue_2153_research_v1_evidence_map.md), [issue_2155_research_v1_ammv_matrix.md](issue_2155_research_v1_ammv_matrix.md) | Claim-gate and protocol note | "A research-v1 matrix and claim gate identify AMV scenario/planner coverage needs." | Do not describe the matrix as executed, statistically sufficient, or paper-ready. | Discussion, Outlook | `future-work only` |
| AMV actuation diagnostics reveal a feasibility-versus-navigation-success split. | [issue_2522_why_first_diagnostics.md](issue_2522_why_first_diagnostics.md), [issue_2443_amv_trace_review.md](issue_2443_amv_trace_review.md), [issue_2440_amv_timeout_closure.md](issue_2440_amv_timeout_closure.md), [issue_2259_amv_clipping_success_boundary.md](issue_2259_amv_clipping_success_boundary.md) | Why-first and trace-review synthesis | "Synthetic actuation-aware scoring can improve command-feasibility signals without moving route progress on the inspected slice." | Do not claim AMV actuation improves navigation, is calibrated to hardware, or generalizes beyond the matched diagnostic row. | Discussion, Limitations | `diagnostic` |
| Static recentering is a local diagnostic mechanism, not a current held-out transfer result. | [issue_2566_static_recenter_inactive_propagation.md](issue_2566_static_recenter_inactive_propagation.md), [issue_2438_static_recenter_activation_closure.md](issue_2438_static_recenter_activation_closure.md), [issue_2182_component_effect_synthesis.md](issue_2182_component_effect_synthesis.md) | Mechanism card and closure synthesis | "Static recentering has local h500 diagnostic support, but the current held-out route is stopped because the mechanism did not activate." | Do not cite static recentering as transferable mitigation or benchmark improvement from the held-out route. | Discussion, Limitations | `diagnostic` |
| Topology guidance is a real diagnostic signal but not a corrective mechanism yet. | [issue_2522_why_first_diagnostics.md](issue_2522_why_first_diagnostics.md), [issue_2530_topology_near_parity_corrective_smoke.md](issue_2530_topology_near_parity_corrective_smoke.md), [issue_2570_topology_revise_status_propagation.md](issue_2570_topology_revise_status_propagation.md), [issue_2563_topology_corrective_revision.md](issue_2563_topology_corrective_revision.md) | Diagnostic smoke and propagation notes | "Near-parity topology selection can leave primary-route dominance, but current evidence still routes the mechanism to revise." | Do not claim topology guidance improves success, transfer, or leaderboard performance before the #2563/#2540 proof path passes. | Discussion, Outlook | `diagnostic` |
| Trace-based mechanism review is useful for explaining failures, but AMMV/AMV trace evidence is still incomplete. | [issue_2236_trace_mechanism_evidence_rubric.md](issue_2236_trace_mechanism_evidence_rubric.md), [issue_2281_research_v1_trace_review_pack.md](issue_2281_research_v1_trace_review_pack.md), [issue_2309_amv_trace_export_blocker.md](issue_2309_amv_trace_export_blocker.md), [mechanism_closure_status.md](mechanism_closure_status.md) | Trace-review rubric and blocker synthesis | "Trace review can separate observed mechanism evidence from hypothesis and identify missing frame/event evidence." | Do not present unavailable AMMV/AMV frame traces as if they exist, and do not infer causality from aggregate-only rows. | Methods, Discussion | `blocked` |
| ScenarioBelief uncertainty and planner projection are promising interface directions. | [issue_1966_scenario_belief_interface.md](issue_1966_scenario_belief_interface.md), [issue_2528_scenario_belief_consumer_smoke.md](issue_2528_scenario_belief_consumer_smoke.md), [issue_2538_scenario_belief_planner_projection.md](issue_2538_scenario_belief_planner_projection.md), [issue_2565_uncertainty_gating_smoke.md](issue_2565_uncertainty_gating_smoke.md) | Interface and smoke notes | "The repository now has sensor-agnostic uncertainty surfaces and smoke-tested projection paths." | Do not claim learned policy performance, uncertainty calibration, or safety improvement from these interface smokes. | Methods, Outlook | `diagnostic` |
| Adversarial and real-trajectory scenario generation are active future-work directions. | [issue_2571_active_research_queue.md](issue_2571_active_research_queue.md), [issue_2468_adversarial_generation_roadmap.md](issue_2468_adversarial_generation_roadmap.md), [issue_2568_adversarial_expansion_gate.md](issue_2568_adversarial_expansion_gate.md), [issue_2523](https://github.com/ll7/robot_sf_ll7/issues/2523) | Queue/roadmap issue surfaces | "Future validation should use bounded adversarial manifests and real-trajectory priors only after the prerequisite artifact gates pass." | Do not imply adversarial expansion, diffusion/RL search, or real-trajectory priors are benchmark evidence today. | Outlook | `future-work only` |
| Learned-policy and learned-risk directions remain blocked on durable data/artifact provenance. | [issue_2225_learned_policy_failure_synthesis.md](issue_2225_learned_policy_failure_synthesis.md), [policy_search/learned_policy_registry.md](policy_search/learned_policy_registry.md), [issue_2273_learned_risk_trace_preflight.md](issue_2273_learned_risk_trace_preflight.md), [issue_2569](https://github.com/ll7/robot_sf_ll7/issues/2569) | Synthesis and launch/preflight notes | "Repeated generic learned-policy continuations are negative or diagnostic; future learned-risk work needs durable trace inputs and artifact provenance." | Do not claim learned methods are ineffective globally, and do not cite launch packets as training or benchmark results. | Limitations, Outlook | `blocked` |

## Unsafe Claim Candidates

- Hardware-calibrated AMV actuation claims remain blocked until source/proxy/hardware provenance
  and runtime fields are accepted.
- CARLA or alternate-simulator transfer claims remain diagnostic or blocked unless native/aligned
  semantics and replay parity evidence are proven for the named slice.
- Open issues, launch packets, proposal notes, and local `output/` files are not dissertation
  evidence by themselves.
- Fallback, degraded, failed, not-available, and accepted-unavailable rows may explain blockers but
  must not be counted as benchmark success.

## Claim Boundary

This report is a compact dissertation-planning index over existing repository evidence. It does not
write dissertation prose, modify any dissertation repository, add benchmark results, or promote
diagnostic findings into manuscript claims.

## Validation

```bash
rtk bash -lc '[ -f docs/context/issue_1542_manuscript_claim_evidence_map.md ]'
rtk bash -lc '[ -f docs/context/issue_2153_research_v1_evidence_map.md ]'
rtk bash -lc '[ -f docs/context/issue_2522_why_first_diagnostics.md ]'
rtk bash -lc '[ -f docs/context/issue_2566_static_recenter_inactive_propagation.md ]'
rtk uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
BASE_REF=origin/main rtk scripts/dev/check_docs_proof_consistency_diff.sh
rtk git diff --check
```
