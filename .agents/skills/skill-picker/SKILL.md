---
name: skill-picker
description: "Choose the most appropriate repo-local skill for an ambiguous task by consulting .agents/skills/README.md."
---

# Skill Picker

## Purpose

Select the smallest useful repo-local skill stack when the user request is ambiguous.

## Workflow

1. Read `.agents/skills/README.md`.
2. Classify request type (issue, PR/review, verification, benchmark, docs, cleanup).
3. Choose one primary skill; add secondary skills only when phases are distinct.
4. If a specific user choice conflicts and is clearly wrong, override with rationale.
5. Report selected skills and continue execution if requested.

## Guardrails

- Do not treat skill-picker as mandatory pre-step for explicit single-skill requests.
- Avoid proposing overly broad bundles that overlap the same phase.
- Use exact repository-specific terminology from AGENTS/skills docs.

## Output

Compact routing note:

- Selected skill(s)
- Why they fit
- Skipped alternatives
- Next action
