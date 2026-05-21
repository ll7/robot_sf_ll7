---
name: auto-improvement
description: "Focused measurement-aware refinement loop for Robot SF prompts, docs, and small code changes; use when a task benefits from trying a few simple improvements."
---

# Auto Improvement

## When to use

Use this skill for small, iterative refinements that should improve clarity, robustness, or validation
coverage without launching full experimentation campaigns.

If the task needs repeated measurable experiments, use `autoresearch` instead.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `.specify/memory/constitution.md`
- `docs/ai/awesome_copilot_adaptation.md`

## Workflow

1. Identify the smallest measurable improvement target.
2. Pick one validation signal and one narrow change at a time.
3. Limit scope to the named files.
4. Apply one tweak; validate; keep or revert quickly.
5. Stop when gain is too small for complexity or the user stops.

## Proof and Guardrails

- Keep changes small and reversible.
- Keep benchmark/provenance wording conservative.
- Prefer `scripts/dev/` wrappers for validation.
- For tasks that are plain cleanup/formatting, use `clean-up`.
- If validation shows no net gain, revert.

## Good Fits

- tightening prompt wording,
- improving docs discoverability,
- reducing duplication in a small helper,
- adding a narrow regression test,
- clarifying a workflow note.

## Output

Summarize:
- what changed,
- what improved,
- what remains open.
