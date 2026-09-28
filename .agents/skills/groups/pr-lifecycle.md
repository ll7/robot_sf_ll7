# PR Lifecycle Skills

Use `gh-pr-comment-fixer` for review feedback, `implementation-verification` for claim proof, `pr-ready-check` for the standard gate, and `gh-pr-opener` for handoff.

## Relationship contract

PR skills read `docs/context/issue_relationships.md` and require the PR's `## Issue Relationship
Mirror` to match the linked issue's fresh native Parent, Blocked by, and Blocking state. Keep
`Closes`/`Refs` in `## Linked Issues` for coverage and closure; they are not graph edges. Do not
infer relationships from branch names, PR prose, or commit messages. Relationship writes belong to
the owning writable implementation/publication worktree; review-only worktrees verify and report
drift but do not create native links. `Relates to` remains manual.
