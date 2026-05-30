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
     value, and oldest issue number,
   - apply status/priority/duration edits in one write pass,
   - run score sync once at batch end if inputs changed.
6. If writes fail (rate limits or auth), stop writes and capture exact pending mutation details.

## Guardrails

- Use MCP/interactive tools when available; use `gh` for deterministic sequencing mutations.
- Do not interleave issue-body edits, project routing, and score sync issue-by-issue for multi-issue batches.
- Use follow-up handoffs rather than retry loops when quotas are temporarily exhausted.

## Output

- Ordered issue queue with one-line rationale per item.
- Status and priority changes applied.
- Unresolved blockers and next candidate issue.
- Whether final score sync completed.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.
