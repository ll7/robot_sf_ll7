# Collision Causal Report Field Map and Decision Record (`collision_causal_report.v1`)

**Status:** proposal / diagnostic-only — schema + fixture contract, not evidence that real
collisions can be causally attributed.
**Issue:** [#5441](https://github.com/ll7/robot_sf_ll7/issues/5441) (parent epic
[#5440](https://github.com/ll7/robot_sf_ll7/issues/5440)).
**Design source:**
[`docs/context/collision_causality_online_risk_scenario_discovery_2026-07-12.md`](collision_causality_online_risk_scenario_discovery_2026-07-12.md)
section 3.
**Owner module:** `robot_sf/benchmark/collision_causal_report.py` +
`robot_sf/benchmark/schemas/collision_causal_report.v1.json`.

Plain-language summary: this contract is a *reviewable incident report* for one simulated
collision. It writes down what was observed, which pipeline stage the proximate cause sits in, and
whether a *named intervention actually prevented contact* — while explicitly refusing to assign
legal or moral fault. It is deliberately model-scoped and fail-closed: any field the repository
cannot yet produce is marked **unavailable** rather than guessed.

## 1. Why a contract first, not a classifier

The parent synthesis (section 3.2.F) requires a reviewable cause report as the first artifact. A
learned cause classifier is premature because the repository does not yet join planner observations,
predictions, candidates, scores, guards, and applied commands into one causal trace, and does not yet
compute the first-unsafe-action or last-avoidable-state timestamps. The honest first increment is a
schema plus fixtures that make those gaps explicit and fail closed.

The contract is **model-scoped** (issue title): planner-internal reconstruction is available only for
planners that emit a decision trace (`mechanism_trace.v1`, currently ORCA-residual). For a black-box
planner the correct report abstains.

## 2. Field producer inventory

Acceptance criterion: *inventory the exact producer for every proposed field and mark unsupported
fields unavailable rather than inferred.* `Availability` describes what the repository can supply
**today**; the schema itself accepts either state per report.

### 2.1 Critical timestamps

| Field | Producer today | Availability | Note |
| --- | --- | --- | --- |
| `t_danger` | `robot_sf/benchmark/critical_intervals.py` (`ttc_threshold_crossing`, `first_braking_event` anchors) | partial | TTC-threshold proxy, not a formal STL-robustness breach. |
| `t_uca` | none | **unavailable** | First-unsafe-action detection is deferred to [#5442](https://github.com/ll7/robot_sf_ll7/issues/5442). |
| `t_inevitable` | none | **unavailable** | Last-avoidable-state / branch search is deferred to #5442. |
| `t_contact` | `robot_sf/benchmark/event_ledger.py` (`CollisionEventRecord.collision_time`) | available | Authoritative collision event. |

### 2.2 Reconstruction elements

| Field | Producer today | Availability | Note |
| --- | --- | --- | --- |
| `observations` | `mechanism_trace.v1.input_condition` | planner-scoped | Only planners emitting a decision trace. |
| `predictions` | `mechanism_trace.v1.risk_score`; forecast batch schema | planner-scoped | No general forecast-distribution join yet. |
| `generated_candidates` | `mechanism_trace.v1.command_source` | planner-scoped | ORCA-residual candidate sources only. |
| `selected_candidate` | `mechanism_trace.v1.selected_command` | planner-scoped | Black-box planners: unavailable. |
| `guard_arbitration_result` | `mechanism_trace.v1.classification` / `command_source` | planner-scoped | |
| `feasible_command` | none (actuation model) | **unavailable** | Kinematic-feasibility projection not exported. |
| `applied_command` | `mechanism_trace.v1.selected_command` | planner-scoped | Proxy for applied command. |
| `actor_states` | `critical_intervals.py` trace arrays (robot/ped positions, velocities) | available | |
| `geometry` | `robot_sf/benchmark/collision_definition_inventory.py` (clearance regime, radii) | available | |

### 2.3 Report-level fields

| Field | Source of truth | Note |
| --- | --- | --- |
| `proximate_mechanism.mechanism_label` | `failure_mechanism_taxonomy.MECHANISM_LABELS` (imported) | Reused verbatim; no competing aggregate taxonomy. |
| `confidence.level` | `failure_mechanism_taxonomy.MECHANISM_CONFIDENCES` (imported) | Reused verbatim. |
| `proximate_mechanism.cause_location` | this contract (`CAUSE_LOCATIONS`) | **Orthogonal** dataflow-stage axis (causal-graph node), *not* a mechanism label. |
| `proximate_mechanism.unsafe_control_action_class` | this contract | STPA four forms + `not_applicable`/`unknown`. |
| `data_source` | `event_ledger` provenance conventions | `source_kind`, `replay_determinism`, `software_commit`. |
| `normative_fault` | const `not_assessed` | Enforced by schema and validator. |

## 3. Temporal causal graph (implemented dataflow)

Acceptance criterion: *document the temporal causal-graph nodes/edges from the implemented
dataflow.* Nodes correspond to `cause_location` values; edges describe the implemented data flow, not
post-hoc statistical correlation.

```
scenario_initialisation
  -> observation_perception        (sensor/obs assembly)
  -> prediction                    (forecast / risk_score)
  -> candidate_generation          (planner candidate set)
  -> candidate_scoring_selection   (selected_command)
  -> safety_guard_arbitration      (guard / classification override)
  -> command_conversion_actuation  (feasible/applied command)
  -> robot_dynamics                (state integration)
  -> pedestrian_response           (closed-loop social force reaction)
  -> collision_metrics             (clearance regime, event ledger)
```

Side nodes not on the main chain: `scenario_infeasibility_or_unavoidable` (contact unavoidable under
the declared action set), `simulator_logging_or_metric_artifact` (spurious/logging cause), and
`unknown_or_interacting` (abstention / interaction set). Edges reflect the ORCA-residual decision
trace ordering; other planners populate a subset and mark the rest unavailable.

## 4. Localization vs intervention-supported actual cause

Acceptance criterion: *explicitly distinguish localization/suspicion from intervention-supported
actual cause.*

- **Localization / suspicion:** `cause_location` + `mechanism_label` + `unsafe_control_action_class`
  say *where* and *how* the incident probably arose. This is derivable from a single observed
  timeline and is **not** proof of cause.
- **Intervention-supported actual cause:** `causal_contribution.supported_actual_cause = true` is
  only legal when a **named** `intervention_model` was run and at least one intervention branch has
  `prevented_contact = true`, the `verdict` is `avoidable`, and `confidence.level` is not `unknown`.
  This mirrors the design's frozen-state branching requirement: causation needs a controlled
  intervention, not a suspicious signal.

Additional fail-closed rules enforced by `collision_causal_report.py`:

- `normative_fault` must be `not_assessed`.
- Unavailable timestamps carry null `step`/`time_s`; unavailable elements carry null
  `source`/`detail`; both must be declared in `missing_fields`.
- If `t_inevitable <= t_uca` (both available) the report may not assert a planner action as the
  supported actual cause — contact was already unavoidable under that model.
- An abstaining report (`abstained = true`) must set `verdict = unknown`,
  `supported_actual_cause = false`, and a non-empty `abstention_reason`.

## 5. Fixtures

- **Complete** (`test_collision_causal_report._complete_report`): a fully reconstructed avoidable
  incident where a frozen-state brake swap prevented contact under a replayed-pedestrian model.
- **Incomplete / abstaining** (`abstained_collision_causal_report`): a black-box planner with no
  decision trace; every reconstruction element and `t_uca`/`t_inevitable` are unavailable, so the
  report abstains instead of fabricating selected/applied actions.

## 6. Decision record

- **Chose** a JSON schema (structural contract) plus a thin Python validator (cross-field fail-closed
  rules) — matching `scenario_contract.py` / `schema_validator.py`, because JSON schema alone cannot
  express the intervention-vs-suspicion and inevitability rules.
- **Reused** `MECHANISM_LABELS` and `MECHANISM_CONFIDENCES` by import (single source of truth) rather
  than copying strings, satisfying "do not create a competing aggregate taxonomy".
- **Added** `cause_location` as an orthogonal dataflow-stage axis rather than extending the aggregate
  outcome taxonomy, because "where in the pipeline" is a different question from "which outcome
  class".
- **Blocked/deferred (honest scope):** `t_uca` and `t_inevitable` computation and general
  cross-planner selected/applied-action provenance do not exist without changing planner behaviour;
  they are out of scope here and owned by #5442. This is the documented stop point from the issue's
  stop rule — the contract fails closed on those fields instead of fabricating them.

## 7. Cross-links

- Parent synthesis: [`collision_causality_online_risk_scenario_discovery_2026-07-12.md`](collision_causality_online_risk_scenario_discovery_2026-07-12.md).
- [#2012](https://github.com/ll7/robot_sf_ll7/issues/2012) — event ledger / typed collision events.
- [#2220](https://github.com/ll7/robot_sf_ll7/issues/2220) — failure-mechanism taxonomy.
- [#2547](https://github.com/ll7/robot_sf_ll7/issues/2547) — critical intervals.
- [#2924](https://github.com/ll7/robot_sf_ll7/issues/2924) — counterfactual pair / intervention pattern.
- [#4758](https://github.com/ll7/robot_sf_ll7/issues/4758) — related trace/provenance work.
- Next: [#5442](https://github.com/ll7/robot_sf_ll7/issues/5442) — snapshot/branching replay and
  last-avoidable-action analysis that will populate `t_uca` / `t_inevitable`.
