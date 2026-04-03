---
name: agentic-eval
description: "Evaluate and improve AI workflow outputs with small goldens, rubrics, and repeatable checks; use when tuning skills, prompts, instructions, or agent behavior."
---

# Agentic Eval

## Overview

Use this skill when the thing being improved is an AI workflow artifact rather than ordinary
application code.

Good targets include skills, instructions, prompt-like text, issue bodies, review rubrics, and
other generated artifacts that benefit from a small evaluation loop.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `docs/ai/awesome_copilot_adaptation.md`
- `code_review.md`

## Workflow

1. Define the artifact
   - Specify what output is being evaluated and what success looks like.
   - Keep the evaluation set small and representative.

2. Build a rubric
   - Use a few concrete criteria.
   - Separate required failures from softer quality preferences.

3. Create a baseline
   - Capture the current output against the rubric before changing anything.

4. Improve one dimension at a time
   - Make a targeted change.
   - Re-run the same evaluation.
   - Keep or discard the change based on the rubric result.

5. Record the result
   - Summarize what improved, what regressed, and what remained hard to judge.

## Guardrails

- If the task has a strong measurable metric and needs repeated experiments, prefer `autoresearch`.
- Keep the evaluation set small enough to rerun quickly.
- Do not overfit to one example at the expense of general usefulness.
