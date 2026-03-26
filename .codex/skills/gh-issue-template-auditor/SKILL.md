---
name: gh-issue-template-auditor
description: "Review existing GitHub issues against the repo's issue-template contract and repair underspecified issues when the fix is clear."
---

# GH Issue Template Auditor

## Overview

Use this skill when an existing issue should be checked against the repo's template contract and
rewritten if it is missing the sections needed for agent-ready execution.

## Read First

- `.github/ISSUE_TEMPLATE/issue_default.md`
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/documentation.md`
- `.github/ISSUE_TEMPLATE/enhancement.md`
- `.github/ISSUE_TEMPLATE/refactor.md`
- `.github/ISSUE_TEMPLATE/research.md`
- `.github/ISSUE_TEMPLATE/planner_integration.md`
- `.github/ISSUE_TEMPLATE/benchmark_experiment.md`
- `scripts/tools/issue_template_audit.py`
- `.codex/skills/gh-issue-clarifier/SKILL.md`

## What to Check

An issue is template-ready only if it includes, at minimum:

- goal or problem statement,
- scope and non-goals,
- added value estimate,
- effort estimate,
- complexity estimate,
- risk assessment,
- affected files or components,
- definition of done,
- success metrics,
- validation or testing,
- project metadata.

## Workflow

1. Inspect the issue
   - `gh issue view <n> --json title,body,labels,milestone,url`
   - Identify which template the issue should have used.

2. Audit the body
   - Use `uv run python scripts/tools/issue_template_audit.py --body-file <body.md>` to identify missing
     sections.
   - If the issue is only missing headings or obvious placeholders, repair it.

3. Repair when safe
   - Rewrite the issue body with the missing sections filled in conservatively.
   - Update the issue with:
     - `gh issue edit <n> --body-file <repaired-body.md>`
   - Keep the original substance intact.

4. Escalate when not safe
   - If the issue is too broad, contradictory, or missing the core problem statement, do not
     guess.
   - Add `decision-required` and move it to `Tracked` if project metadata is in use.
   - Comment with the missing pieces and the narrowest recommended fix.

5. Close the loop
   - If the issue now fits the template contract, say which sections were added or corrected.
   - If it could not be safely repaired, say why and what decision is needed.

## Output Requirements

- Report:
  - issue number,
  - template contract gaps,
  - whether you repaired the body,
  - whether `decision-required` was added,
  - any follow-up recommendation.
