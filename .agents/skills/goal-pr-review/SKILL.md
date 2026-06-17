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
- `scripts/dev/check_skills.py --preflight goal-pr-review` (for preflight validation before review loop)

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

1. Build queue snapshot with `scripts/dev/snapshot_pr_queue.py` (labels, draft status, checks,
   head SHA, last update time) before broad `gh pr view` fields.
2. Sort/prioritize queue (or follow explicit user order).
3. For each PR:
   - capture issue link and head SHA,
   - create or update the active delegation ledger from
     `.agents/skills/goal-autopilot/SKILL.md` with the PR, head SHA, route/run IDs, validation
     plan/status, review/CI state, cleanup status, and next action,
   - run `implementation-verification` for contract alignment,
   - perform an intended-design alignment check before readiness decisions:
     compare the linked issue, design note, PR body, changed behavior, tests, docs, and claims;
     record whether any narrowing was intentional, documented, and still sufficient for the PR,
   - require artifact-first delegated review and validate in order: `result.json`, `RESULT.md`,
     `diffstat.txt`, and `validation.json`, inspect route evidence first, then run targeted local checks
     before raw logs,
   - cap parent-thread raw output at about 200 lines; use `rg -l`, `rg --files`, bounded `sed -n`,
     and private artifacts instead of broad `rg -n .` or full file reads,
   - classify findings as fixable now, deferred, or blocker.

Before choosing the next action for any PR, consult the compact snapshot and apply the
machine-checkable state policy:

```bash
uv run python scripts/dev/pr_loop_policy.py --snapshot <queue-snapshot.json> --json
```

The policy classifies each PR into `pending_ci`, `failed_ci`, `missing_artifacts`,
`stale_worktree`, `ready_to_merge`, or `no_action` and recommends one bounded action
under the loop budget. Use the policy decision to avoid ad-hoc state inspection.

4. Fix actionable items on writable branches; commit and push.
5. Validate per required tier.
6. Re-query unresolved review threads after push and verification before resolving anything, especially
   when moving draft PRs to ready or when bot reviewers were previously pending or skipped.
7. Resolve review threads only after the post-push thread snapshot confirms the fixes still cover all
   actionable comments.
8. Update `merge-ready` only after full proof bar closes.
9. When CI is the only remaining external gate, put the PR in `awaiting_ci` and use compact,
   bounded one-shot polling in non-TTY agent sessions instead of `gh pr checks --watch`:
   `uv run python scripts/dev/watch_pr_ci_status.py <number> --once --json --expected-head-sha <sha>`.
   Inspect JSON/job state first with `gh run view <run-id> --json status,conclusion,jobs` or the
   repo CI helpers. Fetch raw logs only for the relevant failed or completed job, return bounded
   excerpts second with grep/tail, and explicitly label those snippets as bounded excerpts. Keep full logs in private artifacts. Avoid fetching
   `body,comments,reviews,files,statusCheckRollup` together unless the review task explicitly needs
   that full surface. Use `.agents/skills/goal-autopilot/SKILL.md` "Async CI Wait Policy" instead of
   idling the review loop when other safe PR or cycle work remains.
10. Update the active ledger before any CI wait or final handoff. Route completion is not task
   completion until the main agent has verified proof, GitHub state, and cleanup.

## Intended Design And Follow-Up Gate

Before applying `merge-ready`, reviewers must explicitly answer:

- What was the intended design or issue contract?
- Does the implementation behavior match that intent, including tests, docs, and PR claims?
- If the PR intentionally narrowed scope, is the narrowed scope named in the PR or issue and still
  useful on its own?
- Are remaining gaps current-PR blockers, bounded follow-up issues, or handoff-only notes?

Create a follow-up issue when deferred work is real, actionable, and outside the current PR's safe
scope. A good follow-up issue names:

- the residual risk or deferred behavior,
- why it should not block the current PR,
- the acceptance condition or stop rule,
- the expected validation or proof tier,
- links back to the PR, issue contract, design note, or evidence that revealed it.

Block the PR instead of creating a follow-up when the missing work is required for the linked issue
contract, public claim, benchmark interpretation, schema/metric correctness, or safe runtime
behavior. Use a handoff note instead of an issue when the item is only transient state, CI waiting,
local cleanup, or reviewer context with no durable action.

## Proof and Validation

Apply minimum tier by change surface:

- Tier 0: documentation and formatting scope with targeted checks and lint.
- Tier 1: integration and replay changes, CLI runtime smoke, PR readiness.
- Tier 2: planner, metric, scenario, and benchmark behavior.
- Tier 3: campaign-level statistical claims or paper-facing evidence.

`merge-ready` conditions:
- linked issue contract and intended design satisfied, or intentionally narrowed with explicit
  rationale and linked follow-up issues,
- scope matches contract and tests and CI proof are current for reviewed SHA,
- unresolved review threads closed via GitHub review-thread resolution,
- artifacts from `output/` are durably represented or explicitly excluded,
- evidence-producing PRs complete the `Downstream Propagation` section or give an explicit
  not-applicable rationale,
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

## Delegation Failure Recovery

Each child skill or worker may fail. Handle failures per scenario:

- `implementation-verification` failure:
  - If claims are not proven, record specific evidence gaps and leave the PR
    in `under_review`. Do not apply `merge-ready`.
  - If the PR scope does not match the linked issue contract, create a follow-up
    issue and adjust the PR body.

- `pr-ready-check` failure:
  - If lint/format fails, fix and retry once.
  - If tests fail, classify as environmental flake (retry once) or real regression
    (move to `blocked_external` with failure signature).

- `gh-pr-comment-fixer` failure:
  - If push fails after fix, record the error and leave the thread unresolved.
    Move PR to `blocked_external`.
  - If the fix branch has diverged from the remote, skip and report.

- `review-benchmark-change` failure:
  - If benchmark artifacts are missing, record the gap and leave the PR in
    `awaiting_reviewer`. Do not block other PRs.
  - If the benchmark change introduces a regression, report with evidence and
    move the PR to `deferred_scope`.

- `gh-issue-creator` failure:
  - Log the failure and continue. Do not let a follow-up creation failure block
    the PR review.

- `context-note-maintainer` failure:
  - Log the failure and continue. Do not block the PR for a note write failure.

- General environment failure (auth, disk, network):
  - Stop the review loop and report the blocker with the failing command,
    exit code, and minimal next action.

Do not retry a child skill on the same PR if it failed twice with the same
error. Record the recovery action and continue.

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
