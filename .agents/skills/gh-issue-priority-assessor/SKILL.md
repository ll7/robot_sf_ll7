---
name: gh-issue-priority-assessor
description: 'LLM-backed review workflow for Project #5 priority inputs; assess plausibility, propose
  values with uncertainty, route maintainer-value tradeoffs to issue-audit, and optionally apply
  explicit opt-in updates.'
category: github-issue
kind: atomic
phase: context
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# GH Issue Priority Assessor

## Purpose

Assess Project #5 priority fields for plausibility using the canonical rubric, and propose
(or apply, when asked) minimal, evidence-grounded updates. Prefer GitHub MCP / GitHub app tools for
interactive reads when available. Treat numeric scores as sortable approximations from coarse
inputs, not as hard authority.

## Workflow

1. Read rubric and issue context:
   - `docs/project_prioritization.md` (includes the **Research-Leverage Interpretation** and the
     **verify-before-scoring gate** — apply both).
   - `docs/context/issue_713_batch_first_issue_workflow.md` for batch-first Project #5 writes.
   - issue body/metadata from GitHub MCP / GitHub app tools or the canonical complete-thread read
     `uv run python scripts/dev/gh_issue_rest.py thread <number> --repo ll7/robot_sf_ll7`
     (issue #5148: plain `gh issue view --comments` fails on some GitHub CLI versions because it
     requests the deprecated classic-Projects field).
   - current Project #5 field values from MCP or `gh project item-list`.
2. Run the **verify-before-scoring gate**: confirm the issue is still open work (no covering/merged
   PR, no tooling already landed under another issue). If it is already done or superseded, route to
   closure (`issue-audit` / clarifier) instead of scoring it.
3. Evaluate five fields, reading research leverage through them per the rubric (claim-boundary /
   hypothesis → `Improvement`; headline-companion / unblocks-downstream → `Unlock Factor`;
   local-implementable vs SLURM/data-gated → `Success Probability`; deadline → `Time Criticality`):
   - `Improvement`, `Success Probability`, `Time Criticality`, `Unlock Factor`,
     `Expected Duration in Hours`.
4. For each field, state whether current values are plausible, and why.
5. If writeback is requested:
   - apply only fields with enough evidence,
   - use `gh project item-edit` for Project #5 field updates when using the CLI route,
   - batch updates for multiple issues,
   - use `gh project item-edit` when a CLI fallback is needed,
   - run score sync once after the batch.
6. Keep unresolved cases labeled as uncertain with a clear condition that would change the value.
7. When the uncertainty is a maintainer-value tradeoff rather than missing evidence, route to
   `issue-audit` priority discussion instead of inventing a score.

## Auto Mode (empty priorities only)

When invoked by an autonomous loop (e.g. `goal-autopilot`) rather than by an explicit human request,
run in **auto mode**: assess and write back **only issues whose `Priority Score` is currently
empty**, and never recompute or overwrite an existing priority. This keeps each cycle cheap and
leaves human-set priorities stable.

- Discover the empty set with a dry run, then write it in one batch:

  ```bash
  uv run python scripts/tools/project_priority_score.py sync \
    --owner ll7 --project-number 5 --ensure-fields --only-empty --dry-run
  uv run python scripts/tools/project_priority_score.py sync \
    --owner ll7 --project-number 5 --ensure-fields --only-empty \
    --summary-file output/project_priority_score/autofill_summary.json
  ```

- `--only-empty` is the mechanical guard: it skips any item that already has a `Priority Score`.
- Still apply the verify-before-scoring gate to each empty item; an already-merged issue is routed to
  closure, not auto-scored.
- Re-scoring a non-empty priority stays an explicit, human-requested action — auto mode must not do it.

## Guardrails

- Use `docs/project_prioritization.md` as the only scale definition.
- Do not invent new dimensions or Project #5 fields.
- Default to review-only output; writeback only on explicit request **or** in auto mode restricted to
  empty `Priority Score` items (`--only-empty`).
- In auto mode, never overwrite an existing (non-empty) priority.
- Keep issues with contradictions marked for clarifier follow-up.

## Output

- Issue number and current vs proposed field values, including `Priority Score` and
  `Estimate Discussion` when present.
- Field-by-field rationale and uncertainty.
- Plausibility verdict and needed evidence to justify changes, including `Estimate Discussion`
  updates when relevant.
- Whether writeback was applied.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.
