---
name: goal-pr-review
description: "Autonomous goal loop that reviews open PRs, fixes safe actionable gaps, and applies merge-ready after full proof."
---

# Goal PR Review

## Overview

Use this skill when the user wants an autonomous review loop over open PRs. The loop reviews each PR
against its linked issue contract, checks proof quality, fixes safe actionable gaps on writable PR
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
- `.agents/skills/gh-issue-creator/SKILL.md`
- `.agents/skills/context-note-maintainer/SKILL.md`

## Preflight

At the start of each goal, state:

- PR set: all open PRs, non-draft PRs, a label/milestone filter, or explicit PR numbers,
- write mode: PR branch fixes, commits, pushes, PR comments, issue comments, follow-up issue
  creation, thread resolution when appropriate, Project #5 routing, and `merge-ready` labeling are
  allowed by default,
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

Before leaving that comment, first try to fix every failing item that is safely fixable on the PR
branch. A passive "not merge-ready" comment is the fallback only after fixes are impossible, unsafe,
blocked, or still insufficient after validation.

## Fix-First Decision Tree

Classify each unmet requirement before commenting:

- Fixable now:
  - the PR branch is checked out or can be checked out,
  - the branch is writable and can be pushed without force-push or history rewrite,
  - the fix stays inside the linked issue or explicit PR scope,
  - the change is mechanical or evidence-backed, such as lint, formatting, missing docs, missing
    validation note, generated-artifact classification, a small test adjustment, or a clear review
    comment fix.
- Deferred follow-up:
  - the finding is real but outside the current PR contract,
  - the fix would broaden scope or require separate design,
  - the current PR can still satisfy its issue after a dedicated follow-up issue is created with
    `gh-issue-creator`.
- Handoff-only blocker:
  - author intent is unclear,
  - the issue contract is ambiguous,
  - the branch is not writable,
  - required proof depends on unavailable credentials, external services, unavailable CI, heavy
    benchmark infrastructure, or maintainer decisions.

For fixable-now gaps, edit the PR branch, run the narrowest relevant validation, commit, push, and
then reassess the full proof bar. Use `gh-pr-comment-fixer` for review-thread/comment fixes,
`implementation-verification` to map repaired claims back to code, docs, and tests,
`pr-ready-check` when readiness proof is needed, and `gh-issue-creator` for deferred follow-ups.
For missing-proof repairs not tied to review threads, use `implementation-verification` to identify
the smallest code, test, or docs patch, then apply that patch directly on the writable PR branch.
Do not duplicate those skills' detailed procedures here.

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
   - Check whether the PR branch is writable before editing. Prefer repository branches; treat fork
     PRs as review-only unless the branch can be checked out and pushed through normal GitHub
     permissions.
   - Fix every fixable-now gap on the PR branch before leaving a "not merge-ready" comment.
   - If review comments request fixes, use `gh-pr-comment-fixer` when the PR branch is checked out
     and writable.
   - If scope is deferred, create dedicated follow-up issues with `gh-issue-creator` and link them
     from the PR.
   - If evidence is missing but the proof can be produced locally, run the proof and update the PR
     rather than asking the author to do it.
   - If a blocker cannot be fixed safely, comment with the exact blocker and the smallest next
     action needed.

4. Apply or withhold `merge-ready`
   - Apply `merge-ready` only after the full proof bar passes.
   - Remove stale `merge-ready` if a later review finds the proof bar no longer passes.
   - After any fix attempt, commit, push, and record the validation evidence before reassessing the
     label.
   - Leave a short PR comment summarizing the proof basis when applying the label.
   - Leave a "not merge-ready" comment only when attempted fixes are impossible, unsafe, blocked, or
     insufficient after validation.

5. Continue or hand off
   - Continue until the PR queue is exhausted or the stop condition fires.
   - If stopped early, report reviewed PRs, labels changed, blockers, follow-up issues, and next PR.

## Guardrails

- Do not treat passing CI alone as enough for `merge-ready`.
- Do not mark a PR merge-ready if linked issues remain semantically unresolved.
- Do not ignore open review threads; resolve, answer, or convert them to follow-up issues.
- Do not make broad edits just to satisfy readiness; stay inside the issue/PR contract.
- Do not rewrite contributor history or force-push unless the user explicitly authorizes it.
- Do not merge PRs. This skill applies readiness signals only.

## Output Requirements

Report:

- PRs reviewed,
- fix commits pushed,
- `merge-ready` labels applied or removed,
- validation or CI evidence used,
- follow-up issues created,
- unresolved blockers and next recommended review target.
