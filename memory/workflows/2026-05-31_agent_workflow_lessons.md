# Agent Workflow Lessons 2026-05-31

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1849>

## Scope

This memory note stores durable Robot SF workflow lessons that should survive across agent
sessions. It intentionally avoids private rollout logs, prompts, local-only paths, quota details,
and raw delegated-worker transcripts. Treat the entries below as routing and validation guidance,
not as evidence about benchmark or planner quality.

## Durable Lessons

### Re-query live GitHub state before writes

Before changing issue labels, opening PRs, marking work covered, or selecting the next issue, query
the live issue/PR state again. Long goal runs can outlive merges, CI transitions, label cleanup, and
other agents' writes. Live GitHub state plus current `origin/main` should override stale chat
memory, older context notes, and previous queue snapshots.

Useful existing references:

- `docs/context/issue_1776_state_label_routing.md` for state-label routing and stale-label cleanup.
- `docs/context/issue_713_batch_first_issue_workflow.md` for rate-limit-conscious GitHub batching.

### Treat delegated-worker metadata as route health, not proof

A delegated worker reporting success only proves that the route returned a status. Before using a
delegated result as implementation evidence, inspect the returned content for prompt relevance and,
when present, the worker's committed artifact such as `RESULT.md`. Sparse success metadata,
provider summaries, or transport status should not replace local integration review, diff review,
or validation on the branch that will be pushed.

This is especially important when multiple workers run in parallel: the main agent remains
responsible for integration, final judgment, GitHub writes, and PR proof.

### Treat provider rate-limit metadata as no evidence

Provider quota or Copilot rate-limit messages explain why a route could not produce useful work;
they are not evidence that the codebase task is blocked, fixed, or invalid. When a worker route
returns only quota/rate-limit metadata, short-circuit that route, reroute to another capable
worker if the task still matters, or keep the task local. Do not encode the rate-limit incident as
a repository finding unless it affects a committed workflow.

### Bind final PR readiness to committed branch HEAD

Final PR proof should be recorded after the branch is committed, with a clean non-ignored worktree,
and with the readiness stamp tied to the committed `HEAD`. Use:

```bash
PR_READY_MODE=final BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

Plain readiness runs from a dirty worktree are useful interim feedback, but should not be presented
as final PR proof. [Issue #1844](https://github.com/ll7/robot_sf_ll7/issues/1844) added the
final-mode guard; future PR handoffs should cite the committed-head final run when claiming
readiness.

## Follow-Up Boundary

These lessons are stable workflow memory. They do not require importing `.git/codex-agent-runs`
notes, copying private transcripts, or adding new labels. If future runs produce a new recurring
failure mode, add a concise stable lesson here or create a more specific memory note and link it
from `memory/MEMORY.md`.
