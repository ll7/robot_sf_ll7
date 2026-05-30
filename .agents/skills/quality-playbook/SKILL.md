---
name: quality-playbook
description: Repo-wide risk-proportional validation workflow for non-trivial changes with context, risk, validation, and
  follow-through.
category: validation
kind: policy
phase: verification
requires_write: false
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
aliases:
- quality-strategy
- proof-policy
---

# Quality Playbook

## Purpose

Provide a minimal proof-first workflow for higher-risk changes that need stronger validation and
explicit risk accounting without displacing research progress.

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

- Avoid scope expansion and broad refactors during validation-sensitive tasks.
- Do not count fallback/degraded benchmark operation as success evidence.
- Keep follow-up work as explicit backlog items, not silent omissions.

## Output

- Objective + risk class.
- Proof strategy and validation results.
- Residual risk and next required step.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.
