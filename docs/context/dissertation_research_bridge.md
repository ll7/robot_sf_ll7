# Dissertation Research Bridge

Issue: [#2551](https://github.com/ll7/robot_sf_ll7/issues/2551)
Status: dissertation-planning bridge; not dissertation evidence, benchmark proof, safety evidence,
or paper-facing results.

## Purpose

This note maps active Robot SF research lanes to dissertation limitations and future-work
directions without raising the dissertation evidence floor. It is a reading guide over existing
context notes: open issues, proposal notes, diagnostic smokes, and local `output/` artifacts are
not dissertation evidence by themselves.

Use [dissertation_claim_export_candidate_report.md](dissertation_claim_export_candidate_report.md)
for the narrower claim-export candidate list. Use this bridge when explaining how current work
relates to limitations, defense discussion, and future research directions.

Classification vocabulary:

- `current dissertation evidence`: already supported by tracked release/report evidence, with the
  caveats in the cited source.
- `future validation improvement`: an active or proposed proof path that could strengthen the
  dissertation's validation story after new evidence is gathered.
- `future paper direction`: a promising research family that should remain outside the dissertation
  results unless a later paper-grade proof surface exists.
- `defense-prep example`: useful for explaining methodology, negative results, or claim discipline.
- `out of scope`: not suitable for dissertation evidence or wording without a separate project.

## Lane Map

| Lane | Dissertation relevance | Classification | Claim boundary now | Required upstream evidence before manuscript use |
| --- | --- | --- | --- | --- |
| ScenarioBelief uncertainty and stream-gap uncertainty gate | Shows a sensor-agnostic interface direction for uncertainty-aware local planning. | `future validation improvement`; `defense-prep example` | Diagnostic interface and planner-input smoke only; no uncertainty calibration, safety, SNQI, or benchmark improvement claim. | Runtime ScenarioBelief producer or observation-builder path feeding the planner projection, plus durable benchmark or targeted execution evidence with denominators and fail-closed handling. See [issue_2538_scenario_belief_planner_projection.md](issue_2538_scenario_belief_planner_projection.md) and [issue_2565_uncertainty_gating_smoke.md](issue_2565_uncertainty_gating_smoke.md). |
| Adversarial generation | Frames future validation as certified stress-case discovery rather than hand-authored scenario-only testing. | `future paper direction` | Manifest generation and planner smoke are diagnostic; generated cases are not adversarial coverage or planner weakness evidence. | Candidate quality summary, certification/replay determinism, native/comparable planner rows, and promoted benchmark evidence on durable inputs. See [issue_2468_adversarial_generation_roadmap.md](issue_2468_adversarial_generation_roadmap.md), [issue_2524_adversarial_manifests.md](issue_2524_adversarial_manifests.md), and [issue_2562_adversarial_manifest_smoke.md](issue_2562_adversarial_manifest_smoke.md). |
| Counterfactual scenario pairs | Supports a methods/limitations story about mechanism hypotheses and why-first reporting. | `future validation improvement`; `defense-prep example` | Taxonomy/reportability infrastructure only; one pair does not infer causality or benchmark mechanism evidence. | Executed counterfactual pairs with expected-vs-observed metrics, repeated seeds where needed, baseline/intervention denominators, and fallback/degraded exclusions. See [issue_2547_counterfactual_mechanism_taxonomy.md](issue_2547_counterfactual_mechanism_taxonomy.md). |
| Mechanism-aware suites | Connects aggregate benchmark limitations to mechanism-level interpretation. | `future validation improvement`; `defense-prep example` | The #2544 smoke proves suite operability/reportability and records missing trace fields; it is not planner ranking or mechanism proof. | Controlled baseline/intervention slices with required trace fields, row status, durable compact artifacts, and a conservative synthesis surface. See [issue_2452_mechanism_aware_local_nav_suites.md](issue_2452_mechanism_aware_local_nav_suites.md), [issue_2544_mechanism_aware_suite_smoke.md](issue_2544_mechanism_aware_suite_smoke.md), and [mechanism_closure_status.md](mechanism_closure_status.md). |
| AMV actuation | Helps discuss the gap between synthetic robot-command feasibility and navigation success. | `defense-prep example`; limited `current dissertation evidence` only for already tracked AMV scaffold/protocol caveats | Synthetic/proxy diagnostics show feasibility signals and blocked route progress; no calibrated AMV hardware, safety, or planner-superiority claim. | Accepted calibration source or real command-response traces for hardware claims; broader matched benchmark rows for planner claims. See [issue_2456_amv_local_nav_evaluation_suite.md](issue_2456_amv_local_nav_evaluation_suite.md), [issue_2440_amv_timeout_closure.md](issue_2440_amv_timeout_closure.md), and [issue_2224_amv_actuation_ranking.md](issue_2224_amv_actuation_ranking.md). |
| Real-trajectory priors | Explains a future route from synthetic scenarios toward data-informed scenario distributions. | `future paper direction` | Manifest scope only; no raw data staging, learned prior, realism, or performance claim. | License/provenance-approved staged data, importer run, compact scenario-prior manifest under `docs/context/evidence/`, and validation on generated scenario/provenance outputs. See [issue_2479_real_trajectory_scenario_prior.md](issue_2479_real_trajectory_scenario_prior.md). |
| Cyclist interactions | Documents a limitation of pedestrian-only social-navigation evidence and a path toward faster VRU coverage. | `future validation improvement`; `future paper direction` | Cyclist-like VRU fixture is proxy metadata and trace plumbing only; no cyclist dynamics realism or planner ranking. | Native or validated adapter cyclist actor/scenario support, cyclist-specific trace fields and metrics, fail-closed row handling, and one-planner smoke before comparisons. See [issue_2473_cyclist_interaction_benchmark.md](issue_2473_cyclist_interaction_benchmark.md) and [issue_2526_cyclist_vru_smoke.md](issue_2526_cyclist_vru_smoke.md). |
| Signalized crossings | Documents a limitation around right-of-way and traffic-signal semantics. | `future validation improvement`; `future paper direction` | Current signal-state wrapper is trace-only proxy metadata and planner-observable `false`; no legality, forced-waiting, or planner reasoning claim. | Explicit runtime signal phase state, planner-observation policy, zone/legality trace fields, fail-closed row status, and one-planner smoke before comparisons. See [issue_2474_signalized_crossing_benchmark.md](issue_2474_signalized_crossing_benchmark.md) and [issue_2564_signal_state_proxy_smoke.md](issue_2564_signal_state_proxy_smoke.md). |
| SocNavBench / HuNavSim mapping | Positions Robot SF against external social-navigation benchmark and simulator concepts. | `defense-prep example`; `future paper direction` | Literature/interop mapping only; no simulator equivalence, metric parity, scenario transfer, HuNavSim support, or SocNavBench planner-ranking claim. | Fixture-level metric parity or scenario-transfer diagnostics for SocNavBench; new adapter burden/provenance issue before any HuNavSim integration claim. See [issue_2459_socnavbench_hunavsim_mapping.md](issue_2459_socnavbench_hunavsim_mapping.md). |
| Planner mechanism cards | Provides a concise defense aid for why some planner ideas remain diagnostic, revise, stop, or blocked. | `defense-prep example` | Evidence organization only; cards do not add benchmark evidence, promote planners, or make paper-facing claims. | For each card, use the cited canonical smoke command and required diagnostics, then update mechanism-closure status before manuscript wording. See [issue_2453_planner_mechanism_cards.md](issue_2453_planner_mechanism_cards.md) and [research_lane_states.md](research_lane_states.md). |

## Dissertation Wording Guidance

Safe wording should stay at the level of limitations, methods discipline, and future work:

- "Current Robot SF work identifies several validation extensions, including uncertainty-aware
  planner inputs, adversarial scenario manifests, mechanism-aware suites, and richer actor
  semantics."
- "These lanes are tracked as diagnostic or proposal surfaces unless and until they receive
  durable execution evidence, provenance, and fail-closed row classification."
- "Open issues are planning records, not dissertation evidence."

Unsafe wording to avoid:

- Do not say ScenarioBelief uncertainty improves safety or benchmark performance.
- Do not say generated adversarial scenarios prove planner weakness or coverage.
- Do not say cyclist-like VRU metadata is a cyclist simulator.
- Do not say trace-only signal-state metadata proves signal compliance.
- Do not say SocNavBench/HuNavSim mapping establishes simulator equivalence or metric parity.
- Do not cite planner mechanism cards as planner promotion evidence.

## Validation

This bridge is docs-only. Validate by checking referenced paths and the context-doc consistency
diff:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```

## Follow-Up Boundary

Update this note when a lane changes classification in
[research_lane_states.md](research_lane_states.md), when the claim-export report is refreshed, or
when a diagnostic/proposal surface is promoted by durable benchmark or targeted execution evidence.
Do not update dissertation wording from this bridge alone.
