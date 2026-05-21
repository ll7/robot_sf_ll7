---
name: quality-playbook
description: "Repo-wide proof-first workflow for non-trivial changes with context, risk, validation, and follow-through."
---

# Quality Playbook

## Purpose

Provide a minimal, reusable structure for higher-risk changes that need stronger validation and explicit
risk accounting.

## Workflow

1. Restate objective, in-scope boundaries, and affected contracts.
2. Classify risk:
   - behavior, benchmark, docs, reproducibility, AI-workflow.
3. Pick the smallest proving approach:
   - targeted tests, canonical scripts, and representative sample runs.
4. Make the minimal safe change set.
5. Execute checks and document:
   - what passed,
   - what remains uncertain,
   - any required follow-up issue.

## Guardrails

- Avoid scope expansion and broad refactors during proof-first tasks.
- Do not count fallback/degraded operation as success unless explicitly requested.
- Keep follow-up work as explicit backlog items, not silent omissions.

## Output

- Objective + risk class.
- Proof strategy and validation results.
- Residual risk and next required step.
