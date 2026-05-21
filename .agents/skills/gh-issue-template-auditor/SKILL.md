---
name: gh-issue-template-auditor
description: "Review existing GitHub issues against the repo's issue-template contract and repair underspecified issues when the fix is clear."
---

# GH Issue Template Auditor

## Purpose

Check issue bodies against the template contract and perform minimal, safe repairs so issues become
agent-ready without changing intent. Prefer GitHub MCP / GitHub app tools for interactive reads and
writes when available.

## Workflow

1. Read template contract files and `uv run python scripts/tools/issue_template_audit.py`.
2. Load issue body and metadata with `gh issue view`.
3. Compare required sections (problem statement, scope/non-goals, estimates, risks, acceptance, validation, metadata).
4. If gaps are limited and obvious:
   - generate repaired body with missing sections,
   - update with `gh issue edit --body-file`.
5. If gaps are fundamental (unclear objective, wide ambiguity, contradictions):
   - escalate with `decision-required`,
   - keep issue in `Tracked`,
   - add a short clarifying comment.
6. Leave project field writes for the same issue to a separate batching pass.

## Guardrails

- Preserve original issue content and decisions; avoid speculative rewrite.
- Use `decision-required` instead of guessing when scope or problem statement is missing.
- Keep route/metadata cleanup separate from body repair.

## Output

- Issue number and exact missing contract sections.
- Whether repair was applied, skipped, or blocked.
- Any required escalation and follow-up action.
