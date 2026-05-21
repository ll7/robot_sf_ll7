---
name: gh-pr-comment-fixer
description: "Fix GitHub PR review comments with branch-safe edits, validation, and explicit thread resolution."
---

# Gh Pr Comment Fixer

Use this on a writable PR branch when actionable review feedback exists.

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
7. Resolve only addressed threads using the `resolveReviewThread` mutation through
   `gh api graphql`.
8. Re-check thread state; report any unresolved items with blocker reason (permissions, rate limits, or
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
- resolved thread IDs,
- remaining open threads/blockers.
