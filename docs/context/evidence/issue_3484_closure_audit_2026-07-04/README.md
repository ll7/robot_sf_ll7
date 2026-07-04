# Issue 3484 Closure Audit

This audit maps issue
[#3484](https://github.com/ll7/robot_sf_ll7/issues/3484) acceptance criteria to
merged pull requests and retained evidence. Issue #3484 covers feasibility diagnostics for
universally failing `bottleneck`, `cross_trap`, and `head_on_corridor` scenario families.

The audit result is **not closure-ready**: the diagnostic evidence is complete for the current
runner/classifier lanes, but it remains diagnostic-only and does not promote benchmark ranking,
safety-event-ledger, paper-facing, or dissertation-facing claims.

## Acceptance Evidence

| Criterion | Evidence | Status |
| --- | --- | --- |
| Per universally failing family, geometric route-feasibility / clearance verdict. | PR [#4175](https://github.com/ll7/robot_sf_ll7/pull/4175) added the feasibility diagnostic runner. PR [#4389](https://github.com/ll7/robot_sf_ll7/pull/4389) retained the 2026-07-04 evidence packet. `docs/context/evidence/issue_3484_feasibility_diagnostics_2026-07-04/diagnostic_report.json` records `route_feasible=true` for `bottleneck`, `cross_trap`, and `head_on_corridor`. | Delivered as diagnostic evidence. |
| Per family, oracle / scripted successful-trajectory check. | PR [#4175](https://github.com/ll7/robot_sf_ll7/pull/4175) added the scripted diagnostic lane. PR [#4389](https://github.com/ll7/robot_sf_ll7/pull/4389) retained per-family `oracle_solved=false` results. PR [#4441](https://github.com/ll7/robot_sf_ll7/pull/4441) synthesized those retained results into `claim_boundary_synthesis.json`. | Delivered as diagnostic evidence; all families remain non-rankable. |
| Per family, actor-free vehicle-feasibility run. | PR [#4175](https://github.com/ll7/robot_sf_ll7/pull/4175) added actor-free scenario mutation and runner handling. PR [#4389](https://github.com/ll7/robot_sf_ll7/pull/4389) retained results: `actor_free_solved=false` for `bottleneck` and `cross_trap`, `true` for `head_on_corridor`. | Delivered as diagnostic evidence. |
| Per family, extended-time diagnostic. | PR [#4175](https://github.com/ll7/robot_sf_ll7/pull/4175) added optional extended-time lane support. PR [#4389](https://github.com/ll7/robot_sf_ll7/pull/4389) retained an `--include-extended-time` packet with `extended_time_solved=false` for all three target families. | Delivered as diagnostic evidence. |
| Per family, difficulty ramp locates failure boundary. | PR [#4398](https://github.com/ll7/robot_sf_ll7/pull/4398) added diagnostic-only `difficulty_ramp` summary rows. PR [#4441](https://github.com/ll7/robot_sf_ll7/pull/4441) retained the synthesis: `bottleneck` first actor-free/oracle failure at `medium`, `cross_trap` at `low`, and `head_on_corridor` first oracle failure at `medium` with no actor-free failure observed. | Delivered as diagnostic evidence. |
| Diagnostic results emitted into safety-event ledger / benchmark summaries. | PR [#3818](https://github.com/ll7/robot_sf_ll7/pull/3818) added the benchmark-facing compatibility surface `robot_sf/benchmark/scenario_failure_cause.py`. PR [#4389](https://github.com/ll7/robot_sf_ll7/pull/4389) and PR [#4441](https://github.com/ll7/robot_sf_ll7/pull/4441) retained benchmark-reviewable context evidence. No PR in the linked #3484 set updates an automatic safety-event ledger or planner-ranking summary, and issue #3484 itself names safety-event ledger emission as linked work with issue #3482. | Partially delivered; automatic ledger/ranking propagation remains outside this diagnostic-only packet. |
| Universally hard rows reclassified infeasible / time-bound / deadlock / planner-limited with reproducible, versioned verdicts. | PR [#3586](https://github.com/ll7/robot_sf_ll7/pull/3586) introduced `scenario_failure_cause.v1`. PR [#3673](https://github.com/ll7/robot_sf_ll7/pull/3673) tightened evidence schema handling. PR [#3818](https://github.com/ll7/robot_sf_ll7/pull/3818) exposed the classifier through the benchmark namespace. PR [#4441](https://github.com/ll7/robot_sf_ll7/pull/4441) records `bottleneck` and `cross_trap` as `vehicle_infeasible`, `head_on_corridor` as `dynamic_blocking_or_deadlock`, and all three as `comparable_for_ranking=false`. | Delivered as diagnostic evidence; not ranking evidence. |

## Integration Result

The merged PRs answer the examiner-style scenario-validity question at diagnostic strength:

| Family | Route feasible | Actor-free solved | Extended-time solved | Oracle/scripted solved | Diagnostic verdict | Ranking comparable |
| --- | --- | --- | --- | --- | --- | --- |
| `bottleneck` | `true` | `false` | `false` | `false` | `vehicle_infeasible` | `false` |
| `cross_trap` | `true` | `false` | `false` | `false` | `vehicle_infeasible` | `false` |
| `head_on_corridor` | `true` | `true` | `false` | `false` | `dynamic_blocking_or_deadlock` | `false` |

## Closure Boundary

Keep issue #3484 open unless the maintainer explicitly accepts diagnostic-only closure. The
remaining closure blocker is not another guardrail: it is the missing promotion decision for whether
and how these diagnostics should propagate into automatic safety-event-ledger or benchmark-summary
surfaces without creating ranking or paper-facing claims.

Next empirical/actionable step: decide the propagation target. If the target is issue #3484, add a
single ledger/summary integration slice that consumes `scenario_failure_cause.v1` verdicts and
preserves `comparable_for_ranking=false`. If the target is the linked safety-event ledger issue
#3482, leave #3484 open only as a diagnostic evidence dependency and track propagation there.

No full benchmark campaign, Slurm/GPU submission, paper/dissertation claim edit, release, merge, or
deletion was performed for this closure audit.

## Local Validation

```bash
jq -e '.schema_version == "scenario_feasibility_diagnostics.v1" and .claim_boundary == "diagnostic_only_not_benchmark_evidence" and ([.family_verdicts[].failure_cause_verdict.comparable_for_ranking] | all(. == false))' docs/context/evidence/issue_3484_feasibility_diagnostics_2026-07-04/diagnostic_report.json

jq -e '.schema_version == "scenario_feasibility_claim_boundary_synthesis.v1" and .claim_boundary == "diagnostic_only_not_ranking_evidence" and (.still_unsupported_families | length == 0) and (.comparable_for_ranking_families | length == 0)' docs/context/evidence/issue_3484_feasibility_diagnostics_2026-07-04/claim_boundary_synthesis.json

scripts/dev/run_worktree_shared_venv.sh -- uv run --no-sync pytest tests/scenario_certification/test_feasibility_diagnostics.py tests/scenario_certification/test_failure_cause.py tests/benchmark/test_scenario_failure_cause.py -q
```
