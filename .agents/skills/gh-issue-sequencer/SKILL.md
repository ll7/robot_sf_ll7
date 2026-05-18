---
name: gh-issue-sequencer
description: "Maintain a clear next-work queue in GitHub Project #5 by normalizing issue status, priority, and execution order."
---

# GH Issue Sequencer

Use this skill when the user asks to order, triage, or normalize a batch of GitHub issues for
Project #5 execution.

Prefer GitHub MCP / GitHub app tools for interactive issue and project reads when available. Use
`gh` for deterministic batch project writes, score sync, and auth/debugging.

## Read First

- `docs/project_prioritization.md`
- `docs/context/issue_713_batch_first_issue_workflow.md`
- `scripts/tools/project_priority_score.py`
- `.agents/skills/gh-issue-priority-assessor/SKILL.md`
- `.agents/skills/gh-issue-clarifier/SKILL.md`

## Batch-First Rule

Follow `docs/context/issue_713_batch_first_issue_workflow.md`:

1. Do issue text, label, and clarification cleanup first.
2. Do Project #5 field routing second.
3. Run derived score sync once at the end of the batch.
4. Cache project and field IDs once per shell session or in the local Project #5 cache for
   long-running/multi-agent work.
5. Use REST/local `git` for non-project state and reserve GraphQL for Project #5 writes.

Do not interleave body rewrites, status changes, and score sync issue-by-issue unless there is only
one issue in scope.

## Workflow

1. Inspect the queue
   - Check `gh api rate_limit` before large queue work.
   - `gh project item-list 5 --owner ll7 --limit 400 --format json`
   - `gh project field-list 5 --owner ll7 --format json`
   - Use REST (`gh api repos/ll7/robot_sf_ll7/issues?...`) for issue bodies and labels when
     GraphQL quota is low.

2. Identify blockers before sequencing
   - Mark ambiguous issues for `gh-issue-clarifier`.
   - Mark implausible priority inputs for `gh-issue-priority-assessor`.
   - Keep `decision-required`, `blocked`, `duplicate`, and `wontfix` out of the ready queue.

3. Normalize execution status
   - `In progress`: exactly the issue currently being executed.
   - `Ready`: actionable issues with clear scope and validation path.
   - `Tracked`: valid but not ready because they need sequencing, dependencies, or decisions.
   - `Done`: merged/closed work only, unless the team explicitly uses pre-merge done.

4. Order the queue
   - Prefer higher Project #5 priority and lower uncertainty.
   - Prefer issues that unlock downstream work.
   - Keep benchmark/paper-facing blockers ahead of speculative experiments.
   - When priority ties, prefer older issue number first.

5. Apply project writes in one pass
   - Resolve Project #5 and field option IDs once.
   - Use `gh project item-edit` for status/priority/duration writeback.
   - Run `uv run python scripts/tools/project_priority_score.py sync` once after the batch if
     score inputs changed.
   - If GraphQL is exhausted, leave a precise handoff with project ID, item ID, field ID, option ID,
     and the reset time from `gh api rate_limit`.

## Output Requirements

- Report the ordered issue queue with one-line rationale per issue.
- Report any issues moved to `Ready`, `Tracked`, or `In progress`.
- Report whether score sync was run.
- Report unresolved blockers and the next recommended issue to execute.
