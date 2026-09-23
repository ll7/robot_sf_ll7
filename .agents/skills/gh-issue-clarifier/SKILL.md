---
name: gh-issue-clarifier
description: Clarify ambiguous GitHub issues by tightening scope and acceptance criteria, proposing solution
  options with pros/cons, and marking decision-required issues when maintainer input is needed.
category: github-issue
kind: atomic
phase: context
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# GH Issue Clarifier

Compatibility entry point: for new routing, use `issue-contract-maintainer` with mode `clarify-ambiguity` unless a caller explicitly names this skill.


## Purpose

Turn unclear issues into executable tasks with explicit scope, acceptance criteria, and an auditable
decision trail. Keep the issue small, stable, and ready for implementation.

## Workflow

1. Read issue state with the canonical complete-thread read
   `uv run python scripts/dev/gh_issue_rest.py thread <number> --repo ll7/robot_sf_ll7` (issue
   #5148: plain `gh issue view --comments` fails on some GitHub CLI versions because it requests
   the deprecated classic-Projects field) and linked context (PRs, comments, labels, milestone).
   Also read `docs/context/issue_relationships.md` and preserve the canonical `## Relationships`
   block when clarifying the body.
2. Classify ambiguity:
   - Problem, scope, solution, or validation ambiguity.
3. If multiple valid options exist, draft a minimal options set:
   - approach, pros/cons, risk tradeoff, recommendation.
4. Apply the chosen path:
   - clarify issue body in-place (problem, in-scope/out-of-scope, acceptance criteria, tests),
   - if an explicit parent or dependency changes, update the canonical relationship declaration,
     set the native Parent/Blocked by/Blocking link, and read it back; keep `Relates to` manual,
   - only add `decision-required` when maintainer choice is truly needed,
   - create follow-up issues when scope remains too broad.
5. Handle batching discipline:
   - do issue text/label cleanup before Project #5 routing.
   - use `docs/context/issue_713_batch_first_issue_workflow.md` for larger batches.

## Guardrails

- Use MCP for interactive inspection when available; use `gh` as deterministic fallback.
- Do not add assumptions that are not backed by issue context.
- Do not infer native relationships from incidental mentions. Use `none` only when the author or
  maintainer has explicitly established that the relationship does not apply.
- Keep decision comments structured and short (`Context`, `Options`, `Recommendation`, `Decision needed`).
- Remove `decision-required` promptly once decision is recorded.

## Output

- What was ambiguous and the resolved contract.
- Any option analysis and final recommendation.
- Metadata edits (`labels`, project status/milestone where changed).
- Follow-up issues created and reason for each.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.
