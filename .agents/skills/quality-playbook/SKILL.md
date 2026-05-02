---
name: quality-playbook
description: "Repo-wide proof-first workflow for non-trivial changes; use when a task needs context, risk assessment, validation planning, and documented follow-through."
---

# Quality Playbook

## Overview

Use this skill for changes that need more than a quick edit: multi-file fixes, behavior changes,
documentation shifts, or work that could regress a contract if handled casually.

The playbook is intentionally conservative. It focuses on understanding the change surface, choosing
the right proof, and making the result easy to review.

## Read First

- `AGENTS.md`
- `docs/code_review.md`
- `docs/dev_guide.md`
- `.specify/memory/constitution.md`
- `.agent/PLANS.md` for larger tasks

## Workflow

1. Restate the objective
   - Define the change, non-goals, and affected contracts.
   - Note whether the work is code, docs, benchmark, or AI-workflow related.

2. Classify the risk
   - Identify whether the change touches:
     - runtime behavior,
     - benchmark semantics,
     - public docs,
     - reproducibility,
     - AI workflow surfaces.

3. Choose proof
   - Pick the smallest validation that demonstrates the intended contract here.
   - Prefer repo-native scripts, targeted tests, or canonical commands.

4. Make the smallest safe change
   - Keep the scope narrow.
   - Avoid unrelated cleanup unless it is needed to preserve correctness.

5. Validate and document
   - Run the selected checks.
   - Record what passed, what remains uncertain, and any follow-up issue needed.

## Output

Always report objective, risk class, proof chosen, validation result, and residual risk or
follow-up issue.

## Guardrails

- Do not treat fallback or degraded behavior as a success unless the task explicitly measures it.
- Do not widen scope while improving quality.
- Prefer follow-up issues for deferred work instead of silent expansion.
