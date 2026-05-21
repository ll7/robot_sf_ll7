---
name: gh-issue-creator
description: "Create structured GitHub issues from vague prompts using repo templates, conservative assumptions, and Project #5 metadata."
---

# GH Issue Creator

## Purpose

Create GitHub issues from rough prompts with minimal expansion and sufficient structure for agent
execution.

## Workflow

1. Read required inputs:
   - issue templates under `.github/ISSUE_TEMPLATE/`
   - `docs/dev_guide.md`
   - `docs/context/issue_713_batch_first_issue_workflow.md` (for batch routing).
2. Choose the narrowest matching template (`bug`, `enhancement`, `documentation`, `refactor`,
   `research`, `planner_integration.md`, `benchmark_experiment.md`, fallback `issue_default.md`).
3. Normalize prompt into required fields:
   - goal, scope/non-scope, value/effort/complexity/risk, definition of done, success metrics,
     validation plan.
4. Create issue:
   - use GitHub MCP / GitHub app tools when available; use `gh` for deterministic fallback.
   - `gh issue create --title "<title>" --body-file <body.md> --template <template> --label "<labels>"`
   - prefer existing labels; avoid inventing taxonomy.
5. Project routing:
   - `gh project item-add` to add the issue to Project #5 when MCP coverage is unavailable.
   - `gh project item-edit` to set expected fields after resolving project/item/field IDs.
   - add issue to Project #5 and update only expected fields (`Priority`, `Expected Duration in Hours`,
     `Reviewed`) using existing field/schema IDs.
6. Batch discipline:
   - for multiple issues, perform body/label cleanup first, then bulk Project #5 writes, then one sync.
   - if API is rate-limited, record pending project mutations and resume later.

## Guardrails

- Keep assumptions explicit and conservative.
- If template fit is unclear, pick the smallest viable template and note the assumption.
- Do not proceed with speculative follow-up links unless concrete and actionable.
- Use REST for deterministic issue operations; use GraphQL/MCP only where useful for Project #5.

## Output

- Template selected and why.
- Issue URL.
- Final labels, project fields touched, and any batch routing actions queued.
- Explicit assumptions and unresolved metadata updates, if any.
