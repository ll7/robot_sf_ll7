---
name: context-map
description: "Generate a focused repository context map before multi-file changes; use when you need to identify the relevant files, docs, commands, and risks."
---

# Context Map

## When to use

Use this skill before non-trivial work when the main risk is choosing the wrong files or commands.

The objective is a compact map that reduces wrong-surface edits, not execution.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `docs/code_review.md`
- `.specify/memory/constitution.md`
- `.agent/PLANS.md` for larger work

## Workflow

1. Restate the task boundary
   - Capture the requested outcome, likely change type, and any explicit constraints.
   - Separate what is known from what still needs confirmation.

2. Identify the core files
   - List the primary source files, tests, docs, and config files likely to change.
   - Include nearby files that define the same contract or validation path.

3. Identify the validation path
   - Name the repo-native scripts or commands that prove the change.
   - Prefer `scripts/dev/` wrappers and config-first commands.

4. Identify the risk surface
   - Note contracts, benchmarks, reproducibility surfaces, or docs that could drift.
   - Call out any follow-up or provenance concerns.

5. Return a concise map
   - `Primary`
   - `Adjacent`
   - `Validation`
   - `Risks`
   - `Open questions`

## Output

Always include the goal, relevant files/docs, likely commands, risks, and recommended next skill or
execution path.

## Guardrails

- Do not edit files in this skill.
- Do not broaden scope while mapping context.
- If the task is already well-scoped, switch to the appropriate execution skill.

## Proof and Guardrails

- Output should include `Primary`, `Adjacent`, `Validation`, `Risks`, and `Open questions`.
- Do not include speculative recommendations beyond scoped evidence.
