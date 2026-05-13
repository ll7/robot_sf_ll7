---
name: goal-pr-review
description: "Autonomous goal loop that reviews open PRs against issue contracts, repairs safe actionable gaps on writable branches, and applies merge-ready only after full proof."
---

# Goal PR Review

## Overview

Use this skill when the user wants an autonomous review loop over open PRs. The loop reviews each PR
against its linked issue contract, checks proof quality, repairs safe actionable gaps on writable
branches, captures deferred work as follow-up issues, and applies `merge-ready` only when the full
proof bar passes.

This skill reuses `implementation-verification`, `pr-ready-check`, `gh-pr-comment-fixer`,
`review-benchmark-change`, `gh-issue-creator`, and `context-note-maintainer`.

## Read First

- `AGENTS.md`
- `docs/code_review.md`
- `docs/dev_guide.md`
- `docs/context/goal_driven_agent_loops_2026-05-13.md`
- `.github/PULL_REQUEST_TEMPLATE/pr_default.md`
- `.agents/skills/implementation-verification/SKILL.md`
- `.agents/skills/pr-ready-check/SKILL.md`
- `.agents/skills/gh-pr-comment-fixer/SKILL.md`
- `.agents/skills/review-benchmark-change/SKILL.md`

## Preflight

At the start of each goal, state:

- PR set: all open PRs, non-draft PRs, a label/milestone filter, or explicit PR numbers,
- write mode: PR comments, issue comments, follow-up issue creation, thread resolution when
  appropriate, Project #5 routing, and `merge-ready` labeling are allowed by default,
- stop condition: PR queue exhausted, optional time budget reached, validation blocker,
  GitHub/auth failure, or user stop,
- exclusions, such as drafts, PRs by a specific author, benchmark-heavy validation, or external CI
  blockers.

Create the `merge-ready` label if it does not exist before the first successful application.

## Full Proof Bar

Apply `merge-ready` only when all are true:

- the PR links one or more issues or has an explicit standalone scope,
- each linked issue contract is resolved or correctly narrowed,
- the diff matches the stated scope and does not hide unrelated changes,
- tests, CI, or `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` provide adequate proof,
- benchmark/planner changes have benchmark-safe evidence and do not count fallback/degraded
  execution as success,
- unresolved review threads are handled, fixed, or converted into follow-up issues,
- generated `output/` artifacts are classified and durable dependencies are represented,
- deferred but important work has dedicated GitHub issues linked from the PR.

If any item fails, leave a concise review comment or issue/PR follow-up and do not apply
`merge-ready`.

Before leaving a `not merge-ready` outcome, classify each failed proof item with this decision tree:

- auto-fixable now: the PR branch is writable, the fix stays within the linked issue or explicit PR
  contract, the gap is mechanically fixable or a missing-proof repair, and no maintainer decision,
  secret, external service, or heavy unavailable benchmark is required. Repair the branch, validate,
  commit, push, and reassess.
- deferred follow-up: the gap matters but would expand scope beyond the current PR contract. Create
  a dedicated follow-up issue with `gh-issue-creator`, keep the PR scoped, and reassess whether the
  remaining PR can still pass the proof bar.
- handoff-only blocker: the branch is not writable, author intent is ambiguous, the change would
  rewrite contributor history, or proof depends on unavailable external systems or maintainer
  decisions. Leave a concise blocker comment with the exact next action required.

Leave a `not merge-ready` comment only after the auto-fix path is impossible, unsafe, blocked, or
still insufficient after a concrete repair attempt.

## Workflow

1. Build the PR queue
   - Inspect open PRs with labels, draft state, review decision, checks, linked issues, and update
     time.
   - Skip drafts unless the user includes them.
   - Prefer PRs with no `merge-ready` label and older update time first, unless priority is clear.

2. Review one PR at a time
   - Read PR body, linked issues, changed files, CI/checks, review threads, and relevant context
     notes.
   - Use `implementation-verification` to map claims to code, docs, tests, and proof.
   - Use `review-benchmark-change` for benchmark-sensitive PRs.

3. Resolve actionable gaps
   - Classify each failed proof item as auto-fixable now, deferred follow-up, or handoff-only
     blocker before writing comments.
   - Confirm branch writability and scope alignment before editing: the branch must be checked out,
     pushable, and the smallest repair must stay inside the linked issue or explicit PR contract.
   - Use `gh-pr-comment-fixer` for concrete review-thread fixes when the current PR branch is
     writable.
   - Use `implementation-verification` to identify the missing proof surface and the smallest code,
     test, or docs repair that closes it.
   - After every fix attempt, run the narrowest validation that can falsify the repair, then rerun
     the broader readiness proof, usually `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`
     unless a different base or narrower documented proof is required.
   - If ignored or generated `output/` artifacts are involved, classify their persistence before
     pushing and mention the decision in the PR reply.
   - Commit and push each successful repair before reassessing `merge-ready`.
   - If scope is deferred, create dedicated follow-up issues with `gh-issue-creator`.
   - If evidence remains blocked after safe repair attempts, comment with the exact blocker and the
     proof or decision still needed.

4. Apply or withhold `merge-ready`
   - Apply `merge-ready` only after the full proof bar passes.
   - Reassess the full proof bar after each pushed repair; do not rely on pre-fix evidence.
   - Remove stale `merge-ready` if a later review finds the proof bar no longer passes.
   - Leave a short PR comment summarizing the proof basis when applying the label.
   - If one repair lands but another blocker remains, leave a concise outcome comment covering what
     was fixed, which validation passed, and why the PR still is not merge-ready.

5. Continue or hand off
   - Continue until the PR queue is exhausted or the stop condition fires.
   - If stopped early, report reviewed PRs, labels changed, blockers, follow-up issues, and next PR.

## Guardrails

- Do not treat passing CI alone as enough for `merge-ready`.
- Do not mark a PR merge-ready if linked issues remain semantically unresolved.
- Do not ignore open review threads; resolve, answer, or convert them to follow-up issues.
- Do not edit a PR branch unless it is writable and the repair is clearly inside scope.
- Do not leave a passive blocker comment for a safe, scoped, mechanically fixable gap.
- Do not merge PRs. This skill applies readiness signals only.

## Output Requirements

Report:

- PRs reviewed,
- PRs repaired and the pushed commits,
- `merge-ready` labels applied or removed,
- validation or CI evidence used,
- follow-up issues created,
- unresolved blockers and next recommended review target.
