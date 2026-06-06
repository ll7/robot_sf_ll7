---
name: issue-splitter
description: Split a parent, epic, decision, or research issue into the smallest independently implementable
  child issue with duplicate checks and conservative parent linking.
category: github-issue
kind: atomic
phase: planning
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to:
- gh-issue-creator
output_schema: issue_split_summary.v1
aliases:
- parent-to-child-issue
---

# Issue Splitter

## When to use

Use this skill when an open parent, epic, decision, or research issue is too broad to implement
directly, but contains enough context to extract one smallest independently implementable child.
The default path remains one child per run. Use controlled multi-child mode only when the
maintainer explicitly asks for a bounded batch from an already-reviewed split plan or parent issue
set.

Prefer this skill after an implementation queue audit finds no immediately implementable issue, or
when `issue-contract-maintainer` selects `split-parent-to-child` for a parent issue.

Do not use it for automatic broad decomposition of multiple epics, Project #5 field updates, or
creating speculative children that do not have a clear validation path.

## Workflow

1. Read the parent issue body, labels, linked issues/PRs, and recent comments.
2. Identify whether the parent can yield exactly one smallest independently implementable child,
   unless controlled multi-child mode is explicitly authorized.
3. Run a duplicate check before drafting anything:
   - search open and closed issues by parent issue number, distinctive title words, and likely child
     labels,
   - inspect explicit child links in the parent body and recent comments,
   - stop if an equivalent child already exists and report that issue instead.
4. Draft the child in `draft-only` mode by default. Create the issue only when the caller explicitly
   asks for creation or the parent workflow already authorizes GitHub issue writes.
5. Use `gh-issue-creator` for the final issue body shape when creating a child.
6. Update the parent with `Next Implementable Child` only when the relationship is clear and the
   child issue was actually created. Otherwise, return the draft and the exact duplicate check
   query results.

## Controlled Multi-Child Mode

Controlled multi-child mode is disabled by default. Enable it only when all of these are true:

- the maintainer explicitly asks for multiple children or a bounded batch,
- the source is an already-reviewed split plan or parent issue set, not a vague backlog area,
- the batch size is explicit and small, with a default maximum of five children unless the
  maintainer names a smaller limit,
- each candidate has its own duplicate check, `codex_ready_child` contract, scope, non-goals,
  validation path, and `Blocked by` value,
- each created child links the parent, and the parent receives either one batch summary comment or
  body update after creation.

In controlled multi-child mode, do not create a candidate when it is already bounded enough for a
direct PR, cleanly blocked, speculative, a duplicate, or lacking a validation path. Record those
rows as skipped instead of forcing issue creation.

The batch summary must report:

- `created`: children actually created,
- `skipped_duplicate`: candidates skipped because equivalent issues already exist,
- `skipped_blocked`: candidates whose proof path is blocked,
- `skipped_too_broad`: candidates too broad or speculative for a child issue,
- `follow_up`: decisions, parent updates, or maintainer questions left after the batch.

## Child Issue Contract

The child body must include these fields or equivalent headings:

- `Parent issue`: link the parent issue, and name whether the child is extracted from an epic,
  decision, research, or workflow parent.
- `Scope`: the smallest concrete behavior, docs change, validation run, fixture, or analysis that
  can close independently.
- `Non-goals`: parent work deliberately excluded from the child.
- `Validation / Testing`: one command or concrete evidence path that can prove the child.
- `Blocked by`: `none` when the child is ready, or a specific issue, artifact, runtime, maintainer
  decision, or external dependency.

When the child is created, add a concise parent comment or body note:

```markdown
## Next Implementable Child

- <child issue link> - <one-line scope>
```

In controlled multi-child mode, use the plural form and keep the order from the reviewed source
plan:

```markdown
## Next Implementable Children

1. <child issue link> - <one-line scope>
2. <child issue link> - <one-line scope>
```

## Guardrails

- Create at most one child issue per run unless controlled multi-child mode is explicitly
  authorized by the maintainer.
- In controlled multi-child mode, honor the requested batch size, keep the batch small, and skip
  children that would broaden the plan, duplicate existing issues, or hide a blocker.
- Do not mutate Project #5 fields; leave prioritization and status routing to the normal issue
  batching workflow.
- Do not split an issue whose scope is already implementable as one PR; hand it back to
  `goal-issue-implementation`.
- Do not turn a blocked parent into a ready child unless the child's `Blocked by` field is `none`
  and its proof path is available on the current machine.
- Keep duplicate check evidence in the output so future agents can audit why a child was or was not
  created.

## Output

Use `.agents/skills/schemas/issue_split_summary.v1.yaml` as the compact machine-readable
companion to the prose handoff. Keep the prose summary for readability, but include the schema
shape when the caller needs repeatable comparison across runs.

```yaml
issue_split_summary:
  schema: issue_split_summary.v1
  parent_issue: "#..."
  split_mode: single_child | controlled_batch
  mode: draft-only | created | duplicate-found | blocked | batch-draft | batch-created | batch-partial
  batch_request:
    requested: false
    requested_by: "optional maintainer/source"
    source_plan_ref: "optional issue/comment/doc section"
    max_children: 1
    authorized: false
  duplicate_check:
    queries:
      - "..."
    matches:
      - issue: "#..."
        reason: "..."
  proposed_children:
    - title: "..."
      child_order: 1
      issue: "#optional"
      url: "optional"
      readiness: ready_local | blocked_external | blocked_slurm | decision_required | duplicate | too_broad
      batch_status: created | skipped_duplicate | skipped_blocked | skipped_too_broad | deferred | not_applicable
      blocker: none | "..."
      scope: "..."
      non_goals:
        - "..."
      validation:
        - "..."
      duplicate_check:
        queries:
          - "..."
        matches:
          - issue: "#..."
            reason: "..."
  batch_summary:
    requested_children: 0
    created: 0
    skipped_duplicate: 0
    skipped_blocked: 0
    skipped_too_broad: 0
    follow_up:
      - "..."
  recommendation:
    action: create_child | create_batch | use_existing_child | clarify_parent | stop_blocked
    rationale: "..."
  parent_update:
    next_implementable_child: added | skipped
    batch_summary: added | skipped | not_applicable
    note: "..."
```
