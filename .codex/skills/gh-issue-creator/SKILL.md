---
name: gh-issue-creator
description: "Create structured GitHub issues from vague prompts using repo templates, conservative assumptions, and Project #5 metadata."
---

# GH Issue Creator

## Overview

Use this skill when the user wants a new issue created from a rough idea and needs it turned
into a repo-ready issue body with the right template, labels, and project metadata.

Prefer GitHub MCP / GitHub app tools for interactive issue/project inspection when available.
Keep the `gh` commands below as the deterministic fallback for issue creation and batched Project
#5 routing.

## Read First

- `.github/ISSUE_TEMPLATE/issue_default.md`
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/documentation.md`
- `.github/ISSUE_TEMPLATE/enhancement.md`
- `.github/ISSUE_TEMPLATE/refactor.md`
- `.github/ISSUE_TEMPLATE/research.md`
- `.github/ISSUE_TEMPLATE/planner_integration.md`
- `.github/ISSUE_TEMPLATE/benchmark_experiment.md`
- `docs/dev_guide.md`
- `docs/context/issue_713_batch_first_issue_workflow.md`

## Template Selection

Pick the narrowest template that matches the prompt:

- `bug_report.md` for crashes, regressions, incorrect behavior.
- `documentation.md` for docs, guides, or examples.
- `enhancement.md` for product or feature additions.
- `refactor.md` for maintainability-only cleanup.
- `research.md` for experiments or research questions.
- `planner_integration.md` for external or local planner-family integration.
- `benchmark_experiment.md` for benchmark comparisons or scenario work.
- `issue_default.md` when the prompt is vague or mixed.

If multiple templates are plausible, choose the one with the smallest viable scope and say what
assumption you made.

## Workflow

1. Normalize the prompt
   - Extract the concrete goal, scope, likely files, and validation path.
   - Keep assumptions explicit and conservative.

2. Draft the issue body
   - Fill in the selected template.
   - Include:
     - added value estimation,
     - effort estimate,
     - complexity estimate,
     - risk assessment,
     - estimate discussion,
     - definition of done,
     - success metrics,
     - validation/testing.
   - Keep wording tight and executable.

3. Create the issue with `gh`
   - Use the template body as the starting point:
     - `gh issue create --title "<title>" --body-file <body.md> --template <template-name> --label "<labels>"`
   - Prefer labels that already exist in the repository.
   - Avoid inventing new project taxonomy in the issue body.

4. Add project metadata
   - Resolve the project and field IDs first:
     - `gh project view 5 --owner ll7 --format json`
     - `gh project field-list 5 --owner ll7 --format json`
     - `gh project item-list 5 --owner ll7 --limit 200 --format json`
   - Add the issue to Project #5:
     - `gh project item-add 5 --owner ll7 --url <issue-url>`
   - Update existing fields only:
     - `Priority`
     - `Expected Duration in Hours`
     - `Reviewed`
   - Edit the project item with the resolved IDs:
     - `gh project item-edit --id <item-id> --project-id <project-id> --field-id <field-id> --single-select-option-id <option-id>`
     - `gh project item-edit --id <item-id> --project-id <project-id> --field-id <field-id> --number <hours>`
     - `gh project item-edit --id <item-id> --project-id <project-id> --field-id <field-id> --date <YYYY-MM-DD>`
   - Use the current Project #5 schema; do not invent missing fields.
   - When creating many issues, batch the project writes after the issue cleanup pass and run
     score sync once at the end rather than after each issue.

5. Finalize the issue
   - Link dependencies or follow-ups only when they are concrete.
   - Keep the issue open and actionable.

## Output Requirements

- Report:
  - template selected,
  - assumptions made,
  - labels applied,
  - project fields updated,
  - issue URL.
