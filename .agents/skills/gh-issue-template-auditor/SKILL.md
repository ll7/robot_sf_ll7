---
name: gh-issue-template-auditor
description: Review existing GitHub issues against the repo's issue-template contract and repair underspecified
  issues when the fix is clear.
category: github-issue
kind: atomic
phase: context
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# GH Issue Template Auditor

Compatibility entry point: for new routing, use `issue-contract-maintainer` with mode `audit-template-compliance` unless a caller explicitly names this skill.


## Purpose

Check issue bodies against the template contract and perform minimal, safe repairs so issues become
agent-ready without changing intent. Prefer GitHub MCP / GitHub app tools for interactive reads and
writes when available.

## Workflow

1. Read template contract files and run
   `uv run python scripts/tools/issue_template_audit.py` for bulk or targeted audits.
   - For batch routing, preserve `docs/context/issue_713_batch_first_issue_workflow.md`.
   - Preserve the `## Archetype Metadata` YAML block from
    `docs/context/issue_1512_issue_archetypes.md`; repair or flag missing keys and invalid
    `archetype` / `evidence_tier` values conservatively instead of inventing replacements.
2. Load issue body and metadata with GitHub MCP / GitHub app tools when available, or the canonical
   complete-thread read `uv run python scripts/dev/gh_issue_rest.py thread <number> --repo
   ll7/robot_sf_ll7` (issue #5148: plain `gh issue view --comments` fails on some GitHub CLI
   versions because it requests the deprecated classic-Projects field).
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
- Preserve the metadata block even when values are incomplete; flag malformed YAML or invalid
  canonical values instead of deleting or broad-rewriting the issue body.
- Keep route/metadata cleanup separate from body repair.

## Output

- Issue number and exact missing contract sections.
- Whether repair was applied, skipped, or blocked.
- Any required escalation and follow-up action.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.
