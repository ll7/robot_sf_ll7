---
name: goal-pr-review
description: Use for an autonomous Robot SF PR review loop that fixes scoped review gaps, validates proof,
  resolves review threads, and applies merge-ready; not for merging.
category: github-pr
kind: orchestrator
phase: verification
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to:
- implementation-verification
- pr-ready-check
- gh-pr-comment-fixer
- review-benchmark-change
- gh-issue-creator
- context-note-maintainer
output_schema: skill_run_summary.v1
aliases:
- pr-review-runner
---

# Goal PR Review

Use this skill when the user wants a scoped loop over PRs, including fix-safe review actions and
`merge-ready` gating.

It orchestrates:

- `implementation-verification`
- `pr-ready-check`
- `gh-pr-comment-fixer`
- `review-benchmark-change`
- `gh-issue-creator`
- `context-note-maintainer`

Do not let this file absorb subordinate mechanics; keep it as the high-level review contract.

## Trigger Boundary

Use this skill when the user asks to review, fix, verify, or mark open PRs as merge-ready.

Do not use it for:
- implementing new issues from the queue,
- broad repository discovery,
- passive code review where no PR state may be changed,
- merging PRs.

## Read First

- `AGENTS.md`
- `docs/code_review.md`
- `docs/dev_guide.md`
- `docs/context/goal_driven_agent_loops_2026-05-13.md`
- `.github/PULL_REQUEST_TEMPLATE/pr_default.md`
- `.agents/skills/implementation-verification/SKILL.md`
- `.agents/skills/gh-pr-comment-fixer/SKILL.md`
- `.agents/skills/review-benchmark-change/SKILL.md`

## Preflight

Declare at start:
- PR set (all open, non-draft, filtered, or explicit numbers),
- write mode (fix/comment/issue creation/`merge-ready` allowed by default),
- exclusions (drafts, heavy benchmark PRs, external infra blockers),
- stop condition.

Create `merge-ready` label if absent before first successful application.

## State Machine

Each PR is in one state:
- `queued`
- `under_review`
- `fixing`
- `awaiting_ci`
- `awaiting_reviewer`
- `blocked_external`
- `deferred_scope`
- `merge_ready`
- `closed_out`

Avoid loops:
- do not bounce `fixing` ↔ `awaiting_ci` without changes affecting proof.

## Review Workflow

1. Build queue snapshot (labels, draft status, checks, last update time).
2. Sort/prioritize queue (or follow explicit user order).
3. For each PR:
   - capture issue link and head SHA,
   - run `implementation-verification` for contract alignment,
   - classify findings as fixable now, deferred, or blocker.
4. Fix actionable items on writable branches; commit and push.
5. Validate per required tier.
6. Re-query unresolved review threads after push and verification before resolving anything, especially
   when moving draft PRs to ready or when bot reviewers were previously pending or skipped.
7. Resolve review threads only after the post-push thread snapshot confirms the fixes still cover all
   actionable comments.
8. Update `merge-ready` only after full proof bar closes.

## Proof and Validation

Apply minimum tier by change surface:

- Tier 0: documentation and formatting scope with targeted checks and lint.
- Tier 1: integration and replay changes, CLI runtime smoke, PR readiness.
- Tier 2: planner, metric, scenario, and benchmark behavior.
- Tier 3: campaign-level statistical claims or paper-facing evidence.

`merge-ready` conditions:
- linked issue contract satisfied or intentionally narrowed with follow-up issues,
- scope matches contract and tests and CI proof are current for reviewed SHA,
- unresolved review threads closed via GitHub review-thread resolution,
- artifacts from `output/` are durably represented or explicitly excluded,
- benchmark evidence no longer depends on fallback/degraded execution.

If one condition fails, withhold label and emit a blocker comment/follow-up.

## Confidence

Only `High` confidence PRs can receive `merge-ready`.

Confidence meanings:
- `High`: current proof for the reviewed head SHA with closed/blocked threads.
- `Medium`: partial proof or heavy external dependency still open.
- `Low`: missing proof, ambiguous contract, or unavailable environment.

## Anti-Loop and Retry

- Do not rerun the same failing validation twice without code/env change.
- After two repeats, move to `blocked_external` or `awaiting_reviewer` with failure signature and next
  action.
- Do not repeat benchmark campaigns for docs-only or metadata-only PR changes.

## Artifact and Race Rules

- Before final handoff, inspect `output/` locally and classify generated artifacts as:
  - discard
  - ignored-cache
  - evidence-manifest
  - durable-required
- For benchmark-heavy PRs, require scenario set, seed count, and provenance metadata.
- Before pushing/fixing, verify remote PR head has not advanced unexpectedly.
- Avoid force-push and concurrent mutation of the same branch.

## Output Requirements

For each reviewed PR, report:
- PR number, head SHA, queue state transitions,
- validation tier and executed commands,
- fix commits,
- `merge-ready` decision + confidence,
- blockers and `follow-up` issues,
- artifact classification decision.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.


## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.
