# Issue #2220 Failure-Mechanism Taxonomy

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2220>
Date: 2026-06-04
Status: current synthesis over existing diagnostic evidence; no new benchmark run.

## Goal

Define a reusable mechanism-level vocabulary for Robot SF local-navigation failures so planner
comparisons can say what kind of failure is being discussed without overstating causal evidence.
This note is an interpretation layer over existing diagnostics. It does not replace the executable
classifier in [issue_2012_failure_mechanism_classifier.md](issue_2012_failure_mechanism_classifier.md)
or the episode-record taxonomy in
[policy_search/contracts/failure_taxonomy.md](policy_search/contracts/failure_taxonomy.md).

## Confidence Classes

Use one confidence class per mechanism assignment:

| Class | Evidence requirement | Interpretation boundary |
| --- | --- | --- |
| `observed_mechanism` | Trace, replay, video/contact-sheet, direct probe, or explicit root-cause evidence shows the mechanism in the named case. | The case can be cited as observed local evidence, still scoped to its planner/scenario/seed. |
| `supported_hypothesis` | Paired summary rows, component ablations, or compact diagnostics consistently point to the mechanism, but the causal trace is incomplete. | Good for prioritization and follow-up design, not for paper-facing causal claims. |
| `weak_hypothesis` | One seed-local flip, smoke run, or aggregate clue suggests the mechanism. | Use only as a lead for targeted trace evidence. |
| `unknown` | Evidence is unavailable, fallback/degraded, missing, or measures a different interface than the claimed mechanism. | Do not attribute the failure yet. |

Rows from fallback, degraded, failed, missing, partial, or unavailable execution should normally be
`unknown` unless the task is explicitly measuring that unavailable mode.

## Mechanism Vocabulary

| Mechanism | Include when | Exclude when |
| --- | --- | --- |
| `time_budget_artifact` | A stricter horizon times out or remains unfinished, while a longer horizon separates clean completion, exposure-cost completion, or later safety regression. | The longer run is unavailable or no trace/paired-horizon evidence exists. |
| `static_deadlock_or_local_minimum` | Static geometry, bottlenecks, symmetry, or traps produce low progress, indecision, or collision without needing dynamic-agent phase changes. | The dominant effect is route contract invalidity or a moving pedestrian phase. |
| `route_or_topology_mismatch` | Route alternatives, bottleneck choices, route-guide behavior, or pedestrian/robot route offsets change clearance, progress, or failure mode. | Only aggregate terminal metrics change and no route/topology diagnostic is present. |
| `dynamic_phase_or_order_sensitivity` | Pedestrian start time, speed, crossing order, merge gap, or group phase changes closest clearance, progress, or terminal outcome. | The perturbation is spatial only, or trace evidence shows the nearest conflict did not change. |
| `proxemic_or_clearance_tradeoff` | A planner improves success or speed by accepting higher exposure/near-miss/clearance pressure, or improves clearance at the cost of success/runtime. | The only signal is collision/no-collision without exposure or clearance accounting. |
| `guard_or_handoff_domination` | A guard, adapter, observation contract, fallback, or action handoff dominates behavior enough to suppress progress or hide the intended learned/local command. | The guard is merely present and diagnostics show it is not intervening. |
| `learned_policy_low_progress` | Learned or learned-adjacent policies show zero/near-zero success, low progress, or failed continuation without a useful trained contribution signal. | A trained residual or policy contribution is measured and explains the behavior. |
| `actuation_or_command_saturation` | Command projection, AMV/nonholonomic clipping, direct force activation, or adapter constraints materially alter the planner action. | The benchmark path does not expose the actuation-specific metadata needed for attribution. |
| `seed_local_stochastic_fragility` | The same scenario/planner has hard/easy seed partitions or a seed-local outcome flip under otherwise fixed contracts. | The seed difference can already be explained by an observed trace-level mechanism. |
| `scenario_contract_blocker` | Geometry, route clearance, invalid scenario metadata, or infeasible setup makes planner attribution unsafe. | The scenario is certified eligible and the failure recurs under valid execution. |

These labels intentionally describe mechanisms, not raw terminal outcomes. A collision may be
`dynamic_phase_or_order_sensitivity`, `static_deadlock_or_local_minimum`, or
`scenario_contract_blocker` depending on the evidence.

## Mapping To Existing Schemas

- `failure_mechanism_classification.v1` remains the runnable paired-horizon classifier. Its labels
  such as `time_budget_clean_relief`, `exposure_enabled_completion`,
  `safety_regressed_long_horizon`, `persistent_low_progress_timeout`, and
  `scenario_contract_blocker` are diagnostic evidence feeding this taxonomy.
- `policy_search/contracts/failure_taxonomy.md` remains episode-record-driven. Modes such as
  `deadlock`, `timeout_low_progress`, `bottleneck_yield_failure`, and
  `overconservative_stop` should not be promoted to mechanism claims without trace or paired
  evidence.
- Existing `mechanism_hypothesis` fields in seed-mechanism evidence are useful precedent, but
  aggregate-supported rows stay `supported_hypothesis` or weaker until trace/video review confirms
  the causal path.

## Existing Case Map

| Case | Mechanism | Confidence | Evidence and caveat |
| --- | --- | --- | --- |
| ORCA `classic_bottleneck_low`, seed 111, h100 timeout and h500 clean success. | `time_budget_artifact` | `observed_mechanism` | [issue_1056_h500_failure_classification.md](issue_1056_h500_failure_classification.md) cites retained #1049 trace evidence and classifies the row as `time_budget_clean_relief`; this is reporting evidence, not a planner defect. |
| ORCA `classic_t_intersection_medium`, seed 111, h500 completion with higher exposure. | `proxemic_or_clearance_tradeoff` | `observed_mechanism` | [issue_1056_h500_failure_classification.md](issue_1056_h500_failure_classification.md) reports force-exposure increasing from 9 to 50 steps and comfort exposure from 3.0 to 16.667. |
| ORCA `classic_merging_low`, seed 111, h500 collision after fixed-horizon timeout. | `dynamic_phase_or_order_sensitivity` | `observed_mechanism` | [issue_1056_h500_failure_classification.md](issue_1056_h500_failure_classification.md) records collision at step 272 after exposure starts at step 259; recurrence is still needed before broad planner attribution. |
| ORCA issue-596 `narrow_passage`, `symmetry_ambiguous_choice`, and `u_trap_local_minimum` probes. | `static_deadlock_or_local_minimum` | `observed_mechanism` | [issue_596_orca_failure_analysis.md](issue_596_orca_failure_analysis.md) uses contact-sheet and metric probes to identify low-speed stall, symmetry indecision, and U-trap local-minimum behavior. |
| Topology probe on `classic_realworld_double_bottleneck_high`, seed 111. | `route_or_topology_mismatch` | `supported_hypothesis` | [issue_1692_topology_hypothesis_probe.md](issue_1692_topology_hypothesis_probe.md) exposes multiple route hypotheses and selected command sources, but explicitly remains diagnostic-only and not benchmark success evidence. |
| `classic_head_on_corridor_low` pedestrian-route offset and closest-approach trace. | `route_or_topology_mismatch` | `supported_hypothesis` | [issue_1937_ped_route_offset.md](issue_1937_ped_route_offset.md) finds `+0.0978 m` mean min-distance response for pedestrian-route offsets; [issue_1939_corridor_trace_response.md](issue_1939_corridor_trace_response.md) shows `+0.153489 m` closest-clearance delta, with a seed-local progress jump. |
| `francis2023_intersection_wait` start-delay and speed traces. | `dynamic_phase_or_order_sensitivity` | `observed_mechanism` | [issue_1947_intersection_wait_timing_speed_trace.md](issue_1947_intersection_wait_timing_speed_trace.md) records opposite clearance signs for start delay (`+4.159358 m`) and speed offset (`-2.002917 m`) across completed paired traces. |
| Issue #1609 hard/easy seed partitions for crossing, doorway, head-on, merge, and group scenarios. | `seed_local_stochastic_fragility` | `supported_hypothesis` | [mechanism_synthesis_table.csv](evidence/issue_1609_seed_mechanisms_2026-05-31/mechanism_synthesis_table.csv) maps hard/easy seeds to hypotheses such as `crossing_order_timing`, `narrow_passage_timing`, and `merge_gap_timing`, but every row warns that trace/video review is still needed for causal proof. |
| Guarded PPO zero-motion smoke failure and repair. | `guard_or_handoff_domination` | `observed_mechanism` | [issue_2006_guarded_ppo_zero_motion_repair.md](issue_2006_guarded_ppo_zero_motion_repair.md) identifies the wrong observation surface and padded pedestrian handling as root causes; repaired smoke is still single-episode prototype evidence. |
| Learned-policy failure synthesis over BC warm-start PPO, guarded PPO, shielded PPO, ORCA residual, and hybrid-rule comparators. | `learned_policy_low_progress` | `supported_hypothesis` | [issue_2225_learned_policy_failure_synthesis.md](issue_2225_learned_policy_failure_synthesis.md) supports stopping the same BC continuation shape and favoring mechanism-aligned learned interfaces, but it does not claim learned methods are globally ineffective. |
| AMMV-aware Social Force direct mechanism probe versus adapter benchmark rows. | `actuation_or_command_saturation` | `supported_hypothesis` | [issue_2168_ammv_social_force_pair_diagnostic.md](issue_2168_ammv_social_force_pair_diagnostic.md) shows AMMV force activation and lateral-offset effects in the direct probe, while adapter rows are identical and lack AMMV metadata. |
| Adversarial crossing/TTC `failure_0015` and `failure_0045` deterministic replays. | `dynamic_phase_or_order_sensitivity` | `observed_mechanism` | [issue_1861_adversarial_replay_determinism_gate.md](issue_1861_adversarial_replay_determinism_gate.md) reproduces collision signatures for crossing/TTC representatives; mechanism diversity remains limited to that search family. |
| Adversarial head-on route replay with fixed guided route. | `route_or_topology_mismatch` | `observed_mechanism` | [issue_1878_head_on_route_replay_determinism.md](issue_1878_head_on_route_replay_determinism.md) records a deterministic native collision at step 335 from a tracked route fixture; coverage is one seed and one replay row. |

## Storage Recommendation

Keep existing executable fields stable and add mechanism interpretation as optional metadata:

```yaml
mechanism_schema_version: failure_mechanism_taxonomy.v1
mechanism_label: route_or_topology_mismatch
mechanism_confidence: supported_hypothesis
mechanism_evidence_mode: aggregate_summary | paired_trace | deterministic_replay | direct_probe | root_cause | unknown
mechanism_evidence_uri: docs/context/issue_1939_corridor_trace_response.md
mechanism_case_id: issue_1939_classic_head_on_corridor_low_ped_route_offset
mechanism_caveat: seed-local progress jump; not benchmark-strength robustness evidence
```

For benchmark rows, prefer `mechanism_hypothesis` over `failure_mechanism` until trace, replay, or
direct-probe evidence satisfies `observed_mechanism`. For human-reviewed trace cases, storing
`failure_mechanism` is acceptable when the confidence and evidence URI are stored beside it.

## Follow-Up Boundaries

High-value next work should target evidence gaps rather than expanding labels:

- add a compact trace-review artifact for the strongest #1609 aggregate-supported seed partitions;
- crosswalk `failure_mechanism_classification.v1` output into this taxonomy in reporting code only
  after a small fixture demonstrates the mapping;
- require AMMV/AMV benchmark rows to expose actuation-specific metadata before performance
  attribution;
- keep learned-policy rows split between low-progress evidence, handoff/guard domination, and
  unmeasured residual contribution.

Do not use this taxonomy to reclassify old aggregate tables as causal proof. Its main job is to make
future research directions precise enough to choose the next proof step.

## Validation

This synthesis is docs-only over tracked context notes and compact evidence. Validation should use:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
```
