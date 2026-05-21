---
name: what-context-needed
description: "Ask for the minimum repository context needed to answer or implement a task safely."
---

# What Context Needed

## Purpose

Request only the missing inputs required to avoid speculative or unsafe work.

## Workflow

1. Restate the task in one sentence.
2. Group blockers into:
   - missing files/artifacts,
   - missing decisions,
   - missing validation path,
   - ambiguous scope.
3. Ask up to three short questions for precise unblockers (file paths, issue IDs, expected
   command, or decision).
4. Move to execution workflow once missing context is provided.

## Guardrails

- Ask the smallest set of questions that changes uncertainty.
- Do not invent missing facts or default assumptions.
- Do not reuse this skill once required context is available.

## Output

- Minimal missing context checklist and why each item is required.
- The first executable workflow to run next.
