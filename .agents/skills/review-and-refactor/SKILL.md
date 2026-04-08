---
name: review-and-refactor
description: "Surgical review-then-refactor workflow for small code or docs changes; use when a task needs inspection before a narrow improvement."
---

# Review And Refactor

## Overview

Use this skill when you want to inspect a small change surface, identify the real issue, and make a
targeted improvement without turning the task into a broad cleanup.

This is a tighter workflow than a general refactor: review first, then change only what the review
justified.

## Read First

- `AGENTS.md`
- `code_review.md`
- `docs/dev_guide.md`
- `.specify/memory/constitution.md`

## Workflow

1. Inspect the local contract
   - Read the relevant files and nearby tests.
   - Identify the specific behavior or wording that needs adjustment.

2. Review before editing
   - Look for correctness risks, duplicated logic, naming drift, or docs mismatch.
   - Prefer the smallest refactor that resolves the identified issue.

3. Refactor surgically
   - Keep behavior stable unless the task explicitly requires behavior change.
   - Change one concern at a time.

4. Verify the result
   - Run the smallest meaningful validation first.
   - Expand only if the change surface demands it.

## Output

Always report the reviewed surface, the specific issue found, the narrow change made, and the
validation command/result.

## Guardrails

- If the task is pure cleanup, use `clean-up`.
- If the task is benchmark-sensitive, use `review-benchmark-change` or `benchmark-overview` first.
- Do not replace one problem with a wider refactor.
