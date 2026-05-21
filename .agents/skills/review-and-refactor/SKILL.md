---
name: review-and-refactor
description: "Surgical review-then-refactor workflow for small code or docs changes; use when a task needs inspection before a narrow improvement."
---

# Review And Refactor

## Purpose

Avoid broad rewrites: review a small surface first, then surgically apply the smallest safe
refactor needed to fix the identified issue.

## Workflow

1. Read the local contract (AGENTS, docs, related tests) for the target surface.
2. Review for concrete gap (correctness, drift, duplication, naming mismatch).
3. Apply one focused change only.
4. Validate immediately with targeted checks; expand only if needed.

## Guardrails

- Preserve behavior unless the task explicitly requests change.
- Do not switch this into a general cleanup flow.
- For benchmark-sensitive edits, route through benchmark-specific workflows first.

## Output

- Issue found, files touched, and exact narrow fix.
- Validation command(s) and outcome.
- Any risk introduced or deferred.
