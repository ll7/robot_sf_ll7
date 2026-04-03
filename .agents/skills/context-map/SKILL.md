---
name: context-map
description: "Generate a focused repository context map before multi-file changes; use when you need to identify the relevant files, docs, commands, and risks."
---

# Context Map

## Overview

Use this skill before a non-trivial change when the main problem is finding the right repository
surface to inspect.

The goal is not to do the task yet. The goal is to produce a compact map of the files, commands,
and checks that matter most so the next step is low-risk and well-scoped.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `code_review.md`
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

## Guardrails

- Do not edit files in this skill.
- Do not broaden scope while mapping context.
- If the task is already well-scoped, switch to the appropriate execution skill.
