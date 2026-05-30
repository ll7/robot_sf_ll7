---
name: gh-issue-sequencer
description: 'Maintain a clear next-work queue in GitHub Project #5 by normalizing issue status, priority,
  and execution order.'
category: github-issue
kind: atomic
phase: context
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# GH Issue Sequencer

## Purpose

Maintain a clean Project #5 execution queue by separating clarification cleanup, readiness routing,
and one-pass sequencing metadata writes. Project #5 priority is advisory; current maintainer
direction and fresh evidence can override queue order.

## Workflow

1. Prepare:
   - read `docs/project_prioritization.md` and `docs/context/issue_713_batch_first_issue_workflow.md`,
   - resolve project/field IDs once for the session,
   - check `gh api rate_limit` when batch size is large.
2. Inspect queue:
   - list Project #5 items and issue metadata,
   - use REST for issue fields when GraphQL is constrained.
3. Resolve blockers first:
   - route ambiguous issues to `gh-issue-clarifier`,
   - route implausible priorities to `gh-issue-priority-assessor`,
   - keep `decision-required`, `blocked`, `duplicate`, `wontfix` out of execution-ready ordering.
4. Normalize issue status:
   - `In progress` for the active item,
   - `Ready` for actionable and unblocked work,
   - `Tracked` for valid but deferred work,
   - `Done` for merged/closed issues.
5. Order and apply:
   - sort by explicit maintainer direction first, then higher priority, lower uncertainty, unlock
     factor, and oldest issue number,
   - continue autonomous ordering when the top candidate is clearly actionable,
   - use priority discussion only when two or more plausible next issues depend on a real value
     tradeoff that repository evidence and Project #5 fields cannot resolve,
   - apply status/priority/duration edits in one write pass,
   - run score sync once at batch end if inputs changed.
6. If writes fail (rate limits or auth), stop writes and capture exact pending mutation details.

## Priority Discussion Mode

Use this mode sparingly. It exists for queue-ordering tradeoffs, not for routine approval of every
autonomous issue selection.

Ask one focused priority question only when all are true:

- at least two unblocked issues are implementable now,
- their Project #5 score or labels do not clearly decide the order,
- the choice changes real maintainer value, risk, or unblock impact,
- the answer cannot be inferred from recent issue comments, PR state, or canonical docs.

Frame the question around concrete choices, for example:
`Should the next PR prioritize reducing CI minutes (#A) or improving benchmark provenance (#B)?`

After the maintainer answers, record the decision in the relevant issue as a short comment or
`Maintainer priority note` body entry. If the answer changes Project #5 priority fields, batch that
write with the normal Project #5 metadata pass and run score sync once at the end. If the answer is
only an ordering preference, do not invent new priority-score inputs.

## Guardrails

- Use MCP/interactive tools when available; use `gh` for deterministic sequencing mutations.
- Do not interleave issue-body edits, project routing, and score sync issue-by-issue for multi-issue batches.
- Do not ask a priority question when the next issue is already clear, when a blocker/clarification
  question is really needed instead, or when the tradeoff is only agent convenience.
- Use follow-up handoffs rather than retry loops when quotas are temporarily exhausted.

## Output

- Ordered issue queue with one-line rationale per item.
- Priority discussion question asked, answer recorded, or reason it was unnecessary.
- Status and priority changes applied.
- Unresolved blockers and next candidate issue.
- Whether final score sync completed.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.
