# Issue #4164 Closure Audit

Plain-language summary: issue #4164 is not ready to close; merged PRs delivered the Bayesian
goal-posterior module, opt-in planner metadata, and one diagnostic planner-consumption smoke, but
the latest maintainer comment keeps broader benchmark evidence open.

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/4164>
- Audit start issue snapshot: 2026-07-04 22:09 UTC; issue open, four comments read.
- Linked merged PRs verified:
  - PR 4186, `Issue #4164: add Bayesian pedestrian goal inference`, merged 2026-07-02 19:36 UTC.
  - PR 4236, `Issue #4164: wire goal posterior planner input smoke`, merged 2026-07-03 10:46 UTC.
  - PR 4274, `Issue #4164: consume goal posterior in planner comparison`, merged 2026-07-03
    12:18 UTC.

## Claim Boundary

This is a closure-audit integration report only. It does not run a full benchmark campaign, submit
Slurm or GPU compute, change planner or metric semantics, or edit paper or dissertation claims.
The evidence status is `diagnostic-only` for the planner-consumption smoke and `not complete` for
issue closure.

## Acceptance Mapping

| Acceptance criterion or live-thread requirement | Merged evidence | Status |
| --- | --- | --- |
| Implement an interpretable Bayesian goal-inference module with candidate goals, posterior updates, and fail-closed normalization. | PR #4186 added `robot_sf/prediction/goal_intention.py`, public dataclasses/functions, serializable planner summaries, and synthetic tests. | Met by #4186. |
| Unit tests on synthetic trajectories, including goal switch detection within bounded steps. | PR #4186 added `tests/prediction/test_goal_intention.py`; its PR body records `uv run pytest tests/prediction/test_goal_intention.py -q` passed with five tests. | Met by #4186. |
| Candidate-goal source metadata records route-endpoint fallback or blocker state. | PR #4236 extended the goal-posterior planner-input path and smoke evidence. The tracked summary `docs/context/evidence/issue_4164_goal_intention_smoke_summary.json` records route-goal IDs and keeps local smoke output out of git. | Met for diagnostic metadata wiring by #4236. |
| Observation or planner metadata exposes posterior summaries, default disabled. | PR #4236 added `goal_posterior_planner_input_enabled=False`, `goal_posterior_planner_input`, and optional `info["planner_goal_posterior_channel"]` on reset/step. | Met by #4236. |
| One planner path consumes `planner_goal_posterior_channel` during action selection. | PR #4274 added default-off `hybrid_rule_local_planner.goal_posterior_avoidance`, enabled-only command sources, diagnostics, fail-closed blockers, and tests. | Met by #4274. |
| One bounded with/without intent-channel comparison reports command-source changes, trajectory effects, route progress, and fallback/degraded exclusions. | PR #4274 updated `configs/benchmarks/issue_4164_goal_intention_smoke.yaml`, `scripts/benchmark/run_goal_posterior_planner_input_smoke_issue_4164.py`, and focused tests. The PR body classifies this as a CPU smoke proxy, not benchmark-strength evidence. | Met as diagnostic-only smoke evidence by #4274; not issue-closing benchmark evidence. |
| Broader benchmark evidence and any calibrated-intention claim boundary are resolved. | Latest issue comment on 2026-07-03 states: "Remaining under #4164: broader benchmark campaign + calibrated-intention claims. Issue stays open." No merged PR after #4274 provides that campaign or claim-boundary synthesis. | Not met. |

## Closure Decision

Issue #4164 should stay open. The smallest remaining work is no longer another metadata or
guardrail refresh; the next issue-closing slice is a broader benchmark campaign or a maintainer
decision that explicitly narrows closure to the diagnostic smoke evidence already merged.

## Local Verification

Audit-time checks for this docs/state slice:

```bash
gh issue view 4164 --repo ll7/robot_sf_ll7 --comments
gh pr view 4186 --repo ll7/robot_sf_ll7 --comments --json number,title,state,mergedAt,body,comments,files
gh pr view 4236 --repo ll7/robot_sf_ll7 --comments --json number,title,state,mergedAt,body,comments,files
gh pr view 4274 --repo ll7/robot_sf_ll7 --comments --json number,title,state,mergedAt,body,comments,files
```

No full benchmark campaign run, no Slurm or GPU job submitted, no paper or dissertation claim text
changed, no issue comment posted.
