# Issue #2389 Mechanism-Aware Evaluation Thread

Issue: [#2389](https://github.com/ll7/robot_sf_ll7/issues/2389)
Status: current synthesis scaffold; not paper-grade evidence.

## Purpose

This note preserves one explicit publication/dissertation candidate thread for AMV-relevant local
navigation without promoting the current diagnostic and blocked rows beyond their evidence tier.

Candidate argument:

> Aggregate success and collision metrics can hide why local planners fail. Mechanism-aware
> diagnostics can separate transferable improvements from slice-local, inactive, non-corrective,
> blocked, or actuation-feasibility-only effects.

This is a research-routing synthesis. It is not a manuscript section, not a benchmark result, and
not evidence that mechanism-aware evaluation is proven. The current value is that it gives future
work one place to connect mechanism closure state to a possible paper/dissertation argument.

## Source Mechanism Status

| Mechanism lane | Current result | Missing evidence | Dependency / next action | Thread decision |
| --- | --- | --- | --- | --- |
| Static recentering on held-out transfer and static-deadlock rows | Held-out transfer remains `inactive` on the unsolved row: zero activations, unchanged command source, and `0.0 m` trajectory delta. The #2588/#2590 h120 static-deadlock controlled traces each found one active trace-change row (`classic_bottleneck_low`, seed `113`) without terminal rescue; #2592 reran that active row at h500 and both recenter pairings rescued it at step `122`; #2594 repeated across a broader 3x3 h500 slice with the same single active rescue. #2596 classifies this as useful controlled-trace evidence whose promotion is blocked by scope. | No missing evidence for the held-out slice; the mechanism was inactive on the inspected row. For static-deadlock: a harder unsolved-row expansion is needed before promotion. | Reopen held-out only if a future slice predeclares activation-targeted states. For static-deadlock, the next empirical issue should predeclare a harder unsolved-row expansion with the same trace-field contract and a stop rule treating no new unsolved active rescue rows as `synthesize_stop`. | `stop` for held-out transfer; `controlled_trace_negative_mixed` for #2588/#2590; `delayed_rescue_candidate` for #2592; `broader_delayed_rescue_supported` for #2594; `promotion_blocked_by_scope` for #2596 synthesis. |
| Topology guidance / primary-route scoring | Alternatives were generated, but the score surface overselected `primary_route`; the only non-primary selector choice was a numerical tie and did not influence the local command. | A falsification case where a valid non-primary route should clearly win. | Use the near-parity selection preflight before downstream command retuning. | `revise`. |
| AMV actuation-aware scoring | Clipping improved on the tiny AMV timeout trace, yaw saturation did not explain the timeout, and route progress stayed similar. | Route-progress geometry or horizon/task-completion blocker analysis. | Investigate route-progress blockers separately before adding another AMV actuation scorer. | `revise`. |
| AMMV Social Force renderable trace review | The benchmark path regenerates aggregate episode rows, not step-event frames with AMMV force or intrusion metadata. | Durable `simulation_trace_export.v1` step frames for the AMV case. | Implement a recorder/export path or a narrow direct-probe exporter with explicit limitations. | `blocked`. |
| ORCA-residual behavior cloning | The smoke row ran after adapter/JSONL blockers were repaired, avoided collisions and near misses, but timed out with low progress. | A residual objective that explicitly optimizes route progress under the guarded ORCA runtime contract. | Open or execute a revised-objective issue before rerunning the bounded smoke gate. | `revise`. |
| Learned-risk model v1 | Launch-packet checks validate shape and fixture fields, but durable training trace inputs and the baseline artifact URI are missing. | Trace manifest, artifact URI, checksums, labels, and training-readiness proof. | Materialize the inputs or fail closed before training or planner interpretation. | `blocked`. |
| Local learned-policy baseline artifacts | Seven historical local-only model configs are unavailable; scanners now fail closed instead of treating them as promotion-ready. | Durable checkpoint source with checksum and registry/artifact pointer, or explicit config retirement/rewrite. | Do not reuse the same local-only rows for benchmark-facing work. | `stop` until recovered or retired. |

## Interpretation Boundary

Observed evidence supports a useful research direction, but not a paper-ready claim:

- mechanism-aware diagnostics are already useful for deciding whether to stop, revise, or unblock
  specific lanes;
- several rows explain why aggregate metrics alone would be misleading;
- no current row proves a transferable AMV navigation improvement;
- blocked rows mostly identify missing durable inputs, not negative planner behavior;
- diagnostic smoke and trace evidence remain excluded from benchmark-strength claims.

The strongest present claim is methodological and conditional: mechanism-aware evaluation appears
promising as a way to route research effort, but it needs at least one executable mechanism row that
moves from diagnostic or blocked status to controlled candidate evidence before it can support a
paper-facing synthesis.

## Candidate Claim Map

| Candidate thread element | Current support | Required before paper-facing use |
| --- | --- | --- |
| Aggregate metrics hide failure mechanisms. | Diagnostic support from mechanism-aware ranking and mechanism-closure rows. | A paired table that connects aggregate rows, mechanism labels, execution mode, and trace evidence on durable inputs. |
| Some mechanisms should stop early when inactive. | Static recentering inactive trace and local-only learned-policy quarantine. | A documented stop rule tied to predeclared activation conditions across at least one broader slice. |
| Some mechanisms need revision rather than more benchmark runs. | Topology, AMV actuation, and ORCA-residual rows all narrowed a next proof step. | One revised mechanism rerun that uses the predeclared accept/revise/reject gate. |
| Missing evidence should fail closed. | AMMV trace export and learned-risk launch-packet blockers. | Durable recorder/artifact manifests or explicit unavailable classifications before downstream synthesis. |

## Next Empirical Action

Prioritize one executable proof lane before broadening the paper thread:

1. topology near-parity gate rerun, because it has a concrete single-slice falsification plan;
2. AMV step-event trace export, because it would unblock renderable AMMV trace review;
3. ORCA-residual objective revision, because the smoke path already executes but needs a progress
   target;
4. learned-risk trace materialization, only after durable inputs and baseline URI are available.

Do not create a manuscript claim, leaderboard, or broad planner-family comparison from this scaffold
alone.

## Source Links

- Current mechanism closure surface:
  [mechanism_closure_status.md](mechanism_closure_status.md)
- Mechanism-aware ranking diagnostic:
  [issue_2231_mechanism_aware_ranking.md](issue_2231_mechanism_aware_ranking.md)
- Research-v1 claim gate:
  [issue_2153_research_v1_evidence_map.md](issue_2153_research_v1_evidence_map.md)
- Static recentering:
  [issue_2306_static_recenter_activation_trace.md](issue_2306_static_recenter_activation_trace.md),
  [issue_2566_static_recenter_inactive_propagation.md](issue_2566_static_recenter_inactive_propagation.md),
  [issue_2596_static_deadlock_recenter_claim_boundary.md](issue_2596_static_deadlock_recenter_claim_boundary.md)
- Topology scoring:
  [issue_2307_topology_score_diagnostic.md](issue_2307_topology_score_diagnostic.md),
  [issue_2393_topology_selection_preflight.md](issue_2393_topology_selection_preflight.md)
- AMV actuation:
  [issue_2308_amv_timeout_trace_analysis.md](issue_2308_amv_timeout_trace_analysis.md)
- AMMV trace export:
  [issue_2309_amv_trace_export_blocker.md](issue_2309_amv_trace_export_blocker.md)
- ORCA residual:
  [issue_2311_orca_residual_lane_decision.md](issue_2311_orca_residual_lane_decision.md)
- Learned-risk and local learned-policy blockers:
  [issue_2273_learned_risk_trace_preflight.md](issue_2273_learned_risk_trace_preflight.md),
  [issue_2313_local_baseline_quarantine.md](issue_2313_local_baseline_quarantine.md)

## Validation

Docs-only synthesis validation:

```bash
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
