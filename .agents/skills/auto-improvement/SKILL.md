---
name: auto-improvement
description: "Focused measurement-aware refinement loop for Robot SF prompts, docs, and small code changes; use when a task benefits from trying a few simple improvements."
---

# Auto Improvement

## Overview

Use this skill for small, iterative refinement work where the goal is to improve clarity, robustness,
or validation coverage without starting a full experimentation campaign.

If the task has a strong measurable metric and needs repeated experiments, prefer `autoresearch`
instead.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `.specify/memory/constitution.md`
- `docs/ai/awesome_copilot_adaptation.md`

## Workflow

1. Identify the smallest measurable improvement target.
2. Pick one validation signal.
3. Limit scope to the named files.
4. Make one improvement at a time.
5. Keep the change if it improves the signal and stays simple.
6. Revert the change if the gain is too small for the complexity cost.
7. Repeat until the task is good enough or the user says stop.

## Repository Guardrails

- Prefer repo-local scripts and docs as the source of truth.
- Keep benchmark wording conservative.
- Use `scripts/dev/` wrappers when validation is available there.
- If the task is really a plain cleanup or formatting pass, use `clean-up`.

## Good Fits

- tightening prompt wording,
- improving docs discoverability,
- reducing duplication in a small helper,
- adding a narrow regression test,
- clarifying a workflow note.

## Exit

Summarize:
- what changed,
- what improved,
- what remains open.
