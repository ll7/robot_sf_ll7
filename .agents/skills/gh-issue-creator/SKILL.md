---
name: gh-issue-creator
description: 'Create structured GitHub issues from vague prompts using repo templates, conservative assumptions,
  and Project #5 metadata.'
category: github-issue
kind: atomic
phase: context
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# GH Issue Creator

## Purpose

Create GitHub issues from rough prompts with minimal expansion and sufficient structure for agent
execution.

## Workflow

1. Read required inputs:
   - issue templates under `.github/ISSUE_TEMPLATE/`
   - `docs/dev_guide.md`
   - `docs/context/issue_713_batch_first_issue_workflow.md` (for batch routing)
   - `docs/context/issue_1512_issue_archetypes.md` (for archetype and evidence-tier convention)
2. Choose the narrowest matching template (`bug`, `enhancement`, `documentation`, `refactor`,
   `research`, `planner_integration.md`, `benchmark_experiment.md`, fallback
   `issue_default.md`).
3. Normalize prompt into required fields:
   - goal, scope/non-scope, value/effort/complexity/risk, definition of done, success metrics,
     validation plan.
   - for child issues, include `Parent issue`, `Non-goals`, `Validation / Testing`, and
     `Blocked by` fields before creation.
4. Assign an archetype and evidence tier using the convention in
   `docs/context/issue_1512_issue_archetypes.md`. Include the archetype metadata block in
   the issue body. Use only the canonical values from that note. When the archetype is unclear,
   default to `archetype: workflow` with `evidence_tier: idea` and note the assumption.
5. Before creating standalone `blocked-asset` issues, check whether proposed payload
   is only a guardrail, reminder, or "track pasted follow-up anyway" note already
   enforced by existing parent issue `state:blocked` or `evidence:blocked` status.
   If so, do not create new issue; record reminder on parent as comment, label,
   or status update instead. Still create legitimate `blocked-asset` issues when
   they carry unique technical state such as paths, checksums, missing-file
   specifics, asset identifiers, or per-step unblock conditions.
6. Create issue:
   - use GitHub MCP / GitHub app tools when available; use `gh` for deterministic fallback
   - `gh issue create --title "<title>" --body-file <body.md> --template <template> --label "<labels>"`
   - prefer existing labels; avoid inventing taxonomy
   - prefer GitHub MCP / GitHub app tools for interactive issue creation when available; keep `gh`
     for scripted or fallback paths
7. Project routing:
   - use `gh project item-add` when the CLI route is the active Project #5 write path
   - use `gh project item-edit` for explicit field updates when the CLI route is active
   - add issue to Project #5 and update only expected fields (`Priority`, `Expected Duration in Hours`,
     `Reviewed`) using existing field/schema IDs
7. Batch discipline:
   - for multiple issues, perform body/label cleanup first, then bulk Project #5 writes, then one sync
   - if API is rate-limited, record pending project mutations and resume later

## Guardrails

- Keep assumptions explicit and conservative.
- If template fit is unclear, pick the smallest viable template and note the assumption.
- Do not proceed with speculative follow-up links unless concrete and actionable.
- For parent-derived child issues, require a duplicate check from `issue-splitter` or perform one
  before calling `gh issue create`.
- Use REST for deterministic issue operations; use GraphQL/MCP only where useful for Project #5.

## Output

- Template selected and why.
- Issue URL.
- Final labels, project fields touched, and any batch routing actions queued.
- Explicit assumptions and unresolved metadata updates, if any.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.
