<!-- AI-GENERATED: closure-audit integration evidence (PR #5034 follow-up, 2026-07-14) - NEEDS-REVIEW -->

# Issue #5034 — fixed-scope plan and closure audit

Plain-language summary: this bundle preserves the exact CPU-validated launch plan for the control-action-latency sweep so the later authorized operator does not need to rely on ignored worktree output.

This is an integration report for [issue #5034](https://github.com/ll7/robot_sf_ll7/issues/5034). It is launch-plan evidence only: diagnostic-only, not benchmark evidence, not simulator-realism evidence, not sim-to-real evidence, and not a paper-facing claim. The plan enumerates work but launches no simulator episode.

## Source and reproduction

- Audit source commit: `dc90b8987af78309fc6475e358bb2c4fb9960ba1`.
- Configuration: `configs/research/fidelity_sensitivity_v1.yaml`.
- Exact plan command:

  ```bash
  uv run python scripts/benchmark/run_fidelity_sensitivity_campaign.py \
    --fixed-scope-plan-only \
    --require-launchable \
    --plan-out output/fidelity_latency_plan
  ```

- Durable plan summary: `fixed_scope_plan_summary.json`.
- Plan result: `preflight_decision=preflight_ready`, `executable=true`, `launched=false`, and no plan blockers.
- The JSON records the full 48 scenario identifiers. Its resolved scenario source is
  `configs/scenarios/classic_interactions_francis2023.yaml`, requested through
  `configs/benchmarks/paper_experiment_matrix_v1.yaml`.
- Planner groups: `orca`, `default_social_force`, and `hybrid_rule_v0_minimal`.
- Seeds: `111`, `112`, and `113`.
- The plan contains 153 planner × fidelity-axis-variant × seed cells per scenario, including 27 latency cells per scenario for steps `0`, `1`, and `3`; the full resolved scope is 7,344 expected episodes.

The generated plan is reduced here to a small, reviewable scope manifest containing the fields needed to reproduce and gate this issue's fixed scope. The full runner plan remains regenerable by the command above; raw episode JSONL, videos, and campaign output remain excluded from Git.

## Acceptance mapping

| Criterion from #5034 | Status | Evidence and boundary |
| --- | --- | --- |
| A launchable fixed-scope plan is recorded with the exact command, commit, config, scenario set, planners, and seeds. | **Met by this slice** | The durable JSON and this README preserve the current plan and provenance. PR #5026 added the latency axis, PR #5536 resolved the benchmark profile to the runnable 48-scenario source, and PR #5620 made the post-run consumer require exact fixed-scope identities. |
| The native campaign runs without fallback/degraded planner rows, or an exact fail-closed blocker is documented. | **Met via fail-closed branch; native branch pending** | PR #5061 supplied the historical missing-axis blocker; PR #5109 normalizes missing raw rows to compact blocked output; PR #5620 rejects fallback, degraded, malformed, duplicate, missing, and unexpected expected cells. No native episode has run in this audit, so only the permitted blocked alternative—not a native success result—is proven. |
| A durable compact evidence summary links raw artifacts and reports latency metadata plus success, collision, and minimum-clearance metrics for each completed cell. | **Pending real rows** | PR #5085 implemented the promoter and synthetic CPU smoke; PR #5620 added exact plan reconciliation. No native raw rows or real-row summary exist, so synthetic fixture output is not promoted as benchmark evidence. |
| Parent issue #4977 receives the nominal, diagnostic-only, or blocked/not-benchmark-evidence classification. | **Pending native result** | Parent [#4977](https://github.com/ll7/robot_sf_ll7/issues/4977) remains open and has no empirical result classification. Classification must follow the native-row and fallback/degraded disposition. |

## Consolidated contract and remaining work

The merged implementation now has one coherent handoff contract:

1. The plan-only runner resolves and records the fixed scenario/planner/axis/seed scope.
2. The latency preflight requires steps `[0, 1, 3]` before any latency sweep is accepted.
3. The evidence promoter can consume only native rows with structured latency metadata and, in strict mode, exact plan identity coverage; fallback/degraded rows are exclusions, never success evidence.

No new checker or readiness-only packet is introduced here. The new capability is durable publication of the actual launch plan plus this acceptance integration report. The remaining checklist is intentionally empirical:

- [ ] Run the 7,344-episode native fixed-scope campaign on an authorized compute lane.
- [ ] Confirm native/fallback/degraded dispositions and keep excluded rows out of result metrics.
- [ ] Promote real latency rows with the existing strict promoter into the issue-specific evidence bundle and register it.
- [ ] Update parent #4977 with the result classification.

Next empirical action, after an authorized compute decision, is the issue's fixed-scope execution command:

```bash
uv run python scripts/benchmark/run_fidelity_sensitivity_campaign.py \
  --fixed-scope-execute \
  --raw-root output/fidelity_latency_raw \
  --evidence-dir docs/context/evidence
```

The execution remains out of scope for this CPU-only audit. No full benchmark campaign was run, no Slurm/GPU job was submitted, no paper or dissertation claim was edited, and no target host, queue-routing, or packet-lineage state is encoded in this bundle.

## Live audit inputs

- The complete issue thread was read at audit start on 2026-07-14. The latest issue comment then was the merged-PR #5620 remaining-work checklist.
- Substantive merged PRs audited: [#5026](https://github.com/ll7/robot_sf_ll7/pull/5026), [#5061](https://github.com/ll7/robot_sf_ll7/pull/5061), [#5085](https://github.com/ll7/robot_sf_ll7/pull/5085), [#5109](https://github.com/ll7/robot_sf_ll7/pull/5109), [#5536](https://github.com/ll7/robot_sf_ll7/pull/5536), and [#5620](https://github.com/ll7/robot_sf_ll7/pull/5620). Their bodies and review/gate comments consistently leave native execution, real-row promotion, and parent classification open.
- No open PR matched issue #5034 or its exact title during the audit start check. A final live-thread and open-PR recheck is required immediately before PR publication.
