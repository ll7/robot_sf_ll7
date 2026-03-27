---
name: gh-issue-priority-assessor
description: "LLM-backed review workflow for Project #5 priority inputs; assess plausibility, propose values with uncertainty, and optionally apply explicit opt-in updates."
---

# GH Issue Priority Assessor

## Overview

Use this skill when an issue needs a plausibility review for Project #5 priority
inputs or when you want an LLM-backed estimate of the current issue fields.

## Read First

- `docs/project_prioritization.md`
- `docs/context/issue_713_batch_first_issue_workflow.md`
- `.github/ISSUE_TEMPLATE/issue_default.md`
- `.github/ISSUE_TEMPLATE/enhancement.md`
- `.github/ISSUE_TEMPLATE/benchmark_experiment.md`
- `scripts/tools/project_priority_score.py`
- `.codex/skills/gh-issue-template-auditor/SKILL.md`

## Workflow

1. Inspect the issue and current project values
   - `gh issue view <n> --json title,body,labels,milestone,assignees,url`
   - `gh project item-list 5 --owner ll7 --limit 400 --format json`

2. Assess plausibility against the rubric
   - Propose values for:
     - `Improvement`
     - `Success Probability`
     - `Time Criticality`
     - `Unlock Factor`
     - `Expected Duration in Hours`
   - Inspect the issue's `Estimate Discussion` section when present, and update
     it with rationale and uncertainty if writeback is requested.
   - Explain why each value is plausible or not.
   - Note uncertainty and what evidence would move the estimate.
   - Flag contradictions between the issue body and the current values.

3. Keep review-first as the default
   - Output proposed values and rationale without writing fields unless the
     user explicitly asks to apply them.
   - If writeback is requested, update the relevant Project #5 fields with
     `gh project item-edit`.
   - If you are reviewing multiple issues, batch the writebacks and run the
     score sync helper once so `Priority Score` stays derived from the latest
     values.

4. Keep the rubric static
   - Use `docs/project_prioritization.md` as the canonical reference for scales
     and plausibility checks.
   - Do not invent new score dimensions or new Project #5 fields.

## Output Requirements

- Report:
  - issue number,
  - current values,
  - proposed values,
  - rationale and uncertainty,
  - plausibility verdict,
  - whether writeback was applied.
