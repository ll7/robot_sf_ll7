---
name: gh-issue-priority-assessor
description: "LLM-backed review workflow for Project #5 priority inputs; assess plausibility, propose values with uncertainty, and optionally apply explicit opt-in updates."
---

# GH Issue Priority Assessor

## Purpose

Assess Project #5 priority fields for plausibility using the canonical rubric, and propose
(or apply, when asked) minimal, evidence-grounded updates.

## Workflow

1. Read rubric and issue context:
   - `docs/project_prioritization.md`
   - `docs/context/issue_713_batch_first_issue_workflow.md` for batch-first Project #5 writes.
   - issue body/metadata from GitHub MCP / GitHub app tools or `gh issue view`.
   - current Project #5 field values from MCP or `gh project item-list`.
2. Evaluate five fields:
   - `Improvement`, `Success Probability`, `Time Criticality`, `Unlock Factor`,
     `Expected Duration in Hours`.
3. For each field, state whether current values are plausible, and why.
4. If writeback is requested:
   - apply only fields with enough evidence,
   - batch updates for multiple issues,
   - use `gh project item-edit` when a CLI fallback is needed,
   - run score sync once after the batch.
5. Keep unresolved cases labeled as uncertain with a clear condition that would change the value.

## Guardrails

- Use `docs/project_prioritization.md` as the only scale definition.
- Do not invent new dimensions or Project #5 fields.
- Default to review-only output; writeback only on explicit request.
- Keep issues with contradictions marked for clarifier follow-up.

## Output

- Issue number and current vs proposed field values, including `Priority Score` and
  `Estimate Discussion` when present.
- Field-by-field rationale and uncertainty.
- Plausibility verdict and needed evidence to justify changes.
- Whether writeback was applied.
