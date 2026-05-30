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

## Priority Questions

Priority tradeoffs are allowed in this audit loop, but they should not become a confirmation gate
for ordinary autonomous issue pickup. Ask a priority question only when the agent can name the
competing implementable issues and the maintainer's value judgment is the missing input.

Good priority questions are single-choice tradeoffs tied to concrete issue numbers:

- `Which should come first: CI-runtime reduction (#A) or benchmark-provenance cleanup (#B)?`
- `For the next local-only PR, should we favor low-risk docs cleanup (#A) or higher-impact workflow
  repair (#B)?`

After the answer:

- add a short issue comment or `Maintainer priority note` body entry on the issue whose ordering
  changed,
- cite the answer source when editing an issue body,
- update Project #5 priority/status fields only when the answer changes those fields,
- leave Project #5 score inputs advisory and quota-aware,
- check `gh api rate_limit` before Project #5 writeback when applying several priority answers,
  and leave exact pending mutations in the handoff if quota pressure blocks the write,
- return to autonomous sequencing once the specific tradeoff is resolved.

## Guardrails

- Ask questions only for decisions not discoverable from repo artifacts.
- Never ask bundled multi-part questions.
- Do not ask the user to approve every next issue; clear cases should keep moving through the
  normal autonomous implementation loop.
- Do not preserve stale issue statements once a decision invalidates them.

## Output

- Issue and blocker addressed.
- Question asked and answer applied.
- Priority answer recorded and where, if priority discussion mode was used.
- Changes made (body/labels/project state) and follow-up issues.
- Next blocker and optional stop reason if interrupted.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.
