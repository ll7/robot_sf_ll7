# Active Research Lane Scientific States

Issue: [#2464](https://github.com/ll7/robot_sf_ll7/issues/2464)

Status: current governance surface for active research-lane scientific states.

This note classifies research lanes by scientific state, not by GitHub issue status. A row here does
not establish benchmark, paper, transfer, or planner-success evidence. It records the smallest next
discriminating action and the condition under which the lane should stop, revise, or stay blocked.

## State Vocabulary

| State | Meaning |
| --- | --- |
| `candidate` | Plausible mechanism or research family exists, but the discriminating proof is not started. |
| `diagnostic_signal` | A diagnostic signal exists, but it is not strong enough for benchmark promotion. |
| `mechanism_inactive` | The tested mechanism did not visibly activate or did not change the intended field. |
| `active_but_irrelevant` | The mechanism activated, but the measured outcome or failure mode did not improve. |
| `slice_local` | Evidence is useful only for a named scenario, seed, family, or narrow slice. |
| `primary_route_overselected` | Topology or route selection overuses the primary-route explanation. |
| `feasibility_only` | The lane proves a tool, adapter, or diagnostic can run, not that the mechanism helps. |
| `blocked_missing_trace` | The next decision requires trace, frame, activation, or step-event evidence not yet available. |
| `revise` | Existing evidence rejects the current formulation but supports a named revision. |
| `stop` | Existing evidence is sufficient to stop the current lane unless a new hypothesis is opened. |

## Required Row Fields

Each active row must include:

- `lane`
- `issue`
- `current_state`
- `last_evidence_source`
- `next_discriminating_experiment`
- `stop_condition`

Use `scripts/validation/check_research_lane_states.py` to validate the table after edits.

## Active Lane Table

| lane | issue | current_state | last_evidence_source | next_discriminating_experiment | stop_condition |
| --- | --- | --- | --- | --- | --- |
| Topology selection scoring | #2563 | `revise` | [issue_2563_topology_corrective_revision.md](issue_2563_topology_corrective_revision.md) | Try `primary_route_reuse_penalty_under_near_parity_alternatives` in a paired diagnostic that preserves #2530 fields and keeps the gate diagnostic-only. | Stop treating the gate as benchmark improvement evidence until it improves route progress, success, or mechanism evidence beyond the single #2518/#2530 slice. |
| AMV actuation-aware scoring | #2440 | `active_but_irrelevant` | [issue_2440_amv_timeout_closure.md](issue_2440_amv_timeout_closure.md) | Use Issue #2446 only to assess actuation-feasibility as a diagnostic ranking dimension, or switch to route-progress geometry analysis for planner improvement. | Stop broad actuation-aware planner variants unless a new trace shows feasibility changes route progress or success. |
| AMMV/default trace panels | #2434 | `mechanism_inactive` | [issue_2434_ammv_scenario_sweep.md](issue_2434_ammv_scenario_sweep.md) | Add richer AMMV activation instrumentation or choose a deliberately more sensitive diagnostic family. | Stop adapter-mode AMMV behavior-difference claims unless frame-level or episode-level divergence appears. |
| ORCA residual learned lane | #2311 | `revise` | [issue_2311_orca_residual_lane_decision.md](issue_2311_orca_residual_lane_decision.md) | Revise the residual objective or dataset, then rerun the bounded smoke gate before nominal escalation. | Stop the BC lane if a revised smoke still has zero success with timeout-low-progress and no named new objective. |
| Predictive-v2 planner expansion | #2275 | `stop` | [issue_2275_predictive_v2_fate.md](issue_2275_predictive_v2_fate.md) | Open a new planner-coupling or planner-aligned objective hypothesis before any renewed predictive-v2 work. | Stop old predictive-v2 expansion; do not rerun the same planner family unchanged. |
| Static recentering mechanism | #2438 | `mechanism_inactive` | [issue_2438_static_recenter_activation_closure.md](issue_2438_static_recenter_activation_closure.md) | Only continue with a predeclared slice where the static-recenter gate should activate and the same activation fields are recorded. | Stop the current held-out transfer route; do not tune or promote static recentering from the #2221 smoke. |

## Cheap Validation

Run:

```bash
rtk uv run python scripts/validation/check_research_lane_states.py
```

For docs-only edits, also run `rtk git diff --check`. This table should be updated when a new
synthesis changes a lane state, last evidence source, next discriminating experiment, or stop
condition.
