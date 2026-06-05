---
name: issue-contract-maintainer
description: Maintain GitHub issue contracts through template audits, ambiguity clarification, and user-decision
  application.
category: github-issue
kind: orchestrator
phase: planning
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to:
- gh-issue-clarifier
- gh-issue-template-auditor
- issue-audit
- issue-splitter
output_schema: skill_run_summary.v1
aliases:
- issue-contract-audit
- issue-clarification
---

# Issue Contract Maintainer

## When to use

Use this skill when an issue needs contract maintenance before implementation: template compliance, ambiguity clarification, or applying a user decision.

## Modes

- `audit-template-compliance`: compare issue bodies with `.github/ISSUE_TEMPLATE/` and repair clear gaps without changing intent.
- `clarify-ambiguity`: identify problem, scope, solution, or validation ambiguity and post concise options with pros/cons.
- `apply-user-decision`: update issue text, labels, and Project #5 state after the user resolves a readiness question.
- `split-parent-to-child`: hand a parent, epic, decision, or research issue to `issue-splitter`
  when one smallest independently implementable child can be extracted without changing intent.
  Controlled multi-child batches are allowed only when the maintainer explicitly requested a
  bounded batch from a reviewed source plan; otherwise keep the one-child default.

For `audit-template-compliance`, treat the `## Archetype Metadata` YAML block as part of the issue
contract. Preserve the block, validate `archetype` and `evidence_tier` against
`docs/context/issue_1512_issue_archetypes.md`, require a `linked_policy` key to remain present, and
flag malformed YAML or invalid values instead of inventing replacements.

## Workflow

1. Read the issue, labels, linked PRs, recent comments, and Project #5 fields.
2. Choose exactly one mode for the pass and state it before editing.
3. Preserve `docs/context/issue_713_batch_first_issue_workflow.md` batching discipline.
4. Use `gh` REST operations for ordinary issue edits; reserve GraphQL for Project #5 fields.
5. If the fix is not obvious, add or preserve `decision-required`, write the smallest decision prompt, and stop.
6. For `split-parent-to-child`, require a duplicate child check first and update the parent with
   `Next Implementable Child` only after a child issue exists and the relationship is clear.

## Guardrails

- Do not expand the issue beyond the original intent.
- Do not implement the issue from this skill; hand ready work to `gh-issue-autopilot`.
- Do not split a parent into more than one child in a single pass unless the maintainer explicitly
  requested `issue-splitter` controlled multi-child mode for a bounded reviewed plan.
- Ask one readiness-blocking question at a time when user input is needed.
- Keep issue-body edits auditable and mention source comments when applying decisions.

## Output

```yaml
mode: audit-template-compliance | clarify-ambiguity | apply-user-decision | split-parent-to-child
issue: "#..."
edits_made:
  - "..."
project_updates:
  - "..."
blockers:
  - "..."
next_skill: gh-issue-autopilot | gh-issue-sequencer | issue-splitter | none
```
