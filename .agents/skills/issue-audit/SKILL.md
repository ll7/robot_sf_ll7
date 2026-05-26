---
name: issue-audit
description: User-in-the-loop open-issue audit that asks one readiness-blocking question at a time and
  updates issues as decisions are made.
category: github-issue
kind: atomic
phase: context
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# Issue Audit

Compatibility entry point: for new routing, use `issue-contract-maintainer` with mode `apply-user-decision` unless the user explicitly asks for the legacy guided audit loop.


## Purpose

Refine open issues through a guided loop: one blocking question at a time, immediate edits on confirmed
decisions, and clear handoff state.

## Workflow

1. Set scope and ordering:
   - define issue set,
   - start with `decision-required`/contradictory issues,
   - if none, move by Project #5 priority.
2. Gather readiness context:
   - issue states, labels, linked PRs, templates/contracts, and related files.
3. Ask exactly one question at a time focused on scope, acceptance, or priority tradeoff.
4. Apply the decision immediately:
   - edit issue body/labels as needed,
   - remove `decision-required` when resolved,
   - create follow-up issues for bounded splits.
5. Keep batch discipline:
   - issue/body cleanup first,
   - routing/Project #5 updates second,
   - one score-sync batch at the end if needed.

## Guardrails

- Ask questions only for decisions not discoverable from repo artifacts.
- Never ask bundled multi-part questions.
- Do not preserve stale issue statements once a decision invalidates them.

## Output

- Issue and blocker addressed.
- Question asked and answer applied.
- Changes made (body/labels/project state) and follow-up issues.
- Next blocker and optional stop reason if interrupted.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.
