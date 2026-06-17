---
name: gh-pr-comment-fixer
description: Fix GitHub PR review comments with branch-safe edits, validation, and explicit thread resolution.
category: github-pr
kind: atomic
phase: context
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# Gh Pr Comment Fixer

Use this on a writable PR branch when actionable review feedback exists.

If you only need to detect whether review feedback, CI failure, stale head state, or mergeability
needs attention, start with the read-only babysitter snapshot instead:

```bash
uv run python scripts/dev/pr_babysitter_snapshot.py <pr-number> --expected-head-sha <sha> --json
```

Use this skill after that snapshot recommends `process_review_comment` or when the user explicitly
asks to fix known review comments.

## Scope

- Fetch PR context for current branch.
- Collect review threads and top-level PR comments.
- Apply only fixable-now items that stay inside PR scope.
- Run proof at the tightest practical level, then push and resolve threads.

## Workflow

1. Confirm PR for current branch and branch match.
2. Pull requested comments and review threads; keep thread IDs for actionable items.
3. Classify requests:
   - fixable now,
   - needs clarification (ask before editing),
   - out-of-scope (defer).
4. Apply minimal edits grouped by concern.
5. Run appropriate validation (prefer targeted first; full suite only when needed by risk).
6. Commit and push.
7. Re-query unresolved review threads after the push before resolving anything. Bots may add fresh
   findings once a draft PR becomes ready or after the first fix commit.
8. Resolve only addressed threads using the `resolveReviewThread` mutation via `gh api graphql`.
9. Re-check thread state; report any unresolved items with blocker reason (permissions, rate limits, or
   external dependency).

## Anti-Loop / Retry

- Do not rerun unchanged failing validation more than twice without code change or clarified input.
- If a review request cannot be fixed safely, leave a precise blocker note and stop.

## Artifact and Race Rules

- Inspect `output/`-class artifacts before commit and document disposal/durability.
- Verify PR head before applying fixes; avoid resolving threads not yet satisfied by code or commit.
- Do not force-push or mutate thread state for unresolved/ambiguous feedback.

## Required Output

- Which comments were fixed,
- validation command and result,
- commit SHA pushed,
- post-push unresolved-thread re-query result,
- resolved thread IDs,
- remaining open threads/blockers.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.


## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.


## Output

Return the schema named by the `output_schema` frontmatter field, or a compact equivalent when the caller does not require YAML.
