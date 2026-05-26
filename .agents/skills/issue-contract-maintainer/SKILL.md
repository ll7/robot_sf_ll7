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

## Workflow

1. Read the issue, labels, linked PRs, recent comments, and Project #5 fields.
2. Choose exactly one mode for the pass and state it before editing.
3. Preserve `docs/context/issue_713_batch_first_issue_workflow.md` batching discipline.
4. Use `gh` REST operations for ordinary issue edits; reserve GraphQL for Project #5 fields.
5. If the fix is not obvious, add or preserve `decision-required`, write the smallest decision prompt, and stop.

## Guardrails

- Do not expand the issue beyond the original intent.
- Do not implement the issue from this skill; hand ready work to `gh-issue-autopilot`.
- Ask one readiness-blocking question at a time when user input is needed.
- Keep issue-body edits auditable and mention source comments when applying decisions.

## Output

```yaml
mode: audit-template-compliance | clarify-ambiguity | apply-user-decision
issue: "#..."
edits_made:
  - "..."
project_updates:
  - "..."
blockers:
  - "..."
next_skill: gh-issue-autopilot | gh-issue-sequencer | none
```
