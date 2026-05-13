---
name: issue-audit
description: "User-in-the-loop open-issue audit that asks one readiness-blocking question at a time and updates GitHub issues as decisions are made."
---

# Issue Audit

## Overview

Use this skill when the user wants to refine open issues through a discussion loop. This is not a
default `/goal` loop: it keeps the user in the loop, asks one question at a time, and updates issues
as decisions crystallize.

This skill combines the one-question discipline from `grill-me`, the document-grounded challenge
style from `grill-with-docs`, and the repository workflows in `gh-issue-template-auditor`,
`gh-issue-clarifier`, `gh-issue-sequencer`, and `gh-issue-creator`.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `docs/context/goal_driven_agent_loops_2026-05-13.md`
- `docs/context/issue_713_batch_first_issue_workflow.md`
- `.agents/skills/gh-issue-template-auditor/SKILL.md`
- `.agents/skills/gh-issue-clarifier/SKILL.md`
- `.agents/skills/gh-issue-sequencer/SKILL.md`

## Preflight

At the start of the audit, state:

- issue set, such as all open issues, a label, a milestone, a Project #5 slice, or a numbered list,
- write mode: issue edits, comments, labels, project routing, issue splits, and consolidation are
  allowed after each user decision,
- ordering: readiness blockers first,
- stop condition: all selected issues are ready/tracked/blocked with a reason, the user stops, or
  an optional time budget is reached.

## Ordering Policy

Prioritize the next question by readiness impact:

1. `decision-required`, contradictory, or missing core problem statement.
2. Missing acceptance criteria, validation path, or scope/non-goals.
3. Duplicate, consolidation, split, or parent/child ambiguity.
4. `blocked` issues whose unblock condition is unclear.
5. Stale issues whose body conflicts with current repo state.
6. Lower-risk template cleanup and metadata normalization.

Do not spend user attention on already-actionable issues until readiness blockers are exhausted.

## Workflow

1. Gather issue state
   - Inspect open issues and Project #5 items.
   - Run or use `scripts/tools/issue_template_audit.py` where issue body readiness is unclear.
   - Inspect linked PRs, context notes, and relevant files before asking questions that repo
     exploration can answer.

2. Select one blocking question
   - Ask exactly one question at a time.
   - Include the recommended answer and why it is the safest default.
   - The question must materially affect scope, acceptance criteria, sequencing, split/merge
     decisions, blocked state, or validation.

3. Apply the decision immediately
   - Update the issue body, title, labels, comments, and Project #5 routing as needed.
   - Remove `decision-required` once the decision is resolved.
   - Create follow-up issues when the decision splits scope.
   - Mark duplicates or superseded work clearly and link the canonical issue.

4. Maintain batch discipline
   - Do issue text, label, and comment cleanup first.
   - Do Project #5 field routing second.
   - Run derived score sync once at the end of a batch if score inputs changed.

5. Continue or hand off
   - Continue with the next readiness blocker.
   - If stopped early, report the last completed issue, pending question, issues changed, project
     writes still pending, and next recommended issue.

## Guardrails

- Ask the user only about product, priority, scope, or tradeoff decisions that cannot be discovered
  from the repository.
- Do not ask multi-question bundles.
- Do not preserve stale issue text if the decision makes it wrong; update or mark it superseded.
- Do not create broad new issues when a bounded follow-up issue is enough.

## Output Requirements

Report after each decision or audit batch:

- issue number and readiness blocker,
- question asked and selected answer,
- issue/project changes made,
- follow-up or consolidation issues created,
- next readiness blocker.
