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

Prefer this skill after an implementation queue audit finds no immediately implementable issue, or
when `issue-contract-maintainer` selects `split-parent-to-child` for a parent issue.

Do not use it for automatic broad decomposition of multiple epics, Project #5 field updates, or
creating speculative children that do not have a clear validation path.

## Workflow

1. Read the parent issue body, labels, linked issues/PRs, and recent comments.
2. Identify whether the parent can yield exactly one smallest independently implementable child.
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

## Guardrails

- Create at most one child issue per run.
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
  mode: draft-only | created | duplicate-found | blocked
  duplicate_check:
    queries:
      - "..."
    matches:
      - issue: "#..."
        reason: "..."
  proposed_children:
    - title: "..."
      issue: "#optional"
      url: "optional"
      readiness: ready_local | blocked_external | blocked_slurm | decision_required | duplicate | too_broad
      blocker: none | "..."
      scope: "..."
      non_goals:
        - "..."
      validation:
        - "..."
  recommendation:
    action: create_child | use_existing_child | clarify_parent | stop_blocked
    rationale: "..."
  parent_update:
    next_implementable_child: added | skipped
    note: "..."
```
