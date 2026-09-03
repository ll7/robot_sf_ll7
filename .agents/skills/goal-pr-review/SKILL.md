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

Always read:

- `AGENTS.md`
- `docs/code_review.md`
- `docs/dev_guide.md`
- `docs/context/goal_driven_agent_loops_2026-05-13.md`
- `.github/PULL_REQUEST_TEMPLATE/pr_default.md`
- `.agents/skills/implementation-verification/SKILL.md`
- `scripts/dev/check_skills.py --preflight goal-pr-review` (for preflight validation before review loop)

Read when applicable:

- `.agents/skills/gh-pr-comment-fixer/SKILL.md` (only when a PR has unresolved review threads to fix)
- `.agents/skills/review-benchmark-change/SKILL.md` (only for benchmark-facing PRs)

## Preflight

Declare at start:
- PR set (all open, non-draft, filtered, or explicit numbers),
- write mode (fix, label, comment, branch/PR update, issue creation, thread resolution, and
  `merge-ready` are allowed by default),
- exclusions (drafts, heavy benchmark PRs, external infra blockers),
- stop condition.

Create `merge-ready` label if absent before first successful application.

Before diagnosing any individual PR, run the shared-main baseline check (see
`## Shared-Main Baseline Before Per-PR Diagnosis`). A red baseline on `origin/main` changes how
every PR's CI failure is classified for the rest of the run.

## Factory Authority And Label Source Of Truth

Once the owner or parent orchestrator starts a bounded review run, review each in-scope PR without
per-PR confirmation. The reviewer may apply labels, post comments, push safe fixes to writable
branches, update PR bodies, create successor issues, and resolve threads whose findings are proven
fixed. `gh-pr-merger` retains merge authority.

Treat live GitHub labels and PR state as authoritative across machines. Projects, cached queue
snapshots, local ledgers, and worker reports are evidence or mirrors only; they never override the
current label set, head SHA, or GitHub thread state.

Owner approval is required before deleting durable scientific artifacts, experiment records or
runs, or GitHub releases. Routine scratch cleanup and deletion of a fully preserved merged feature
branch do not waive that durable-deletion boundary.

## Concurrent Writers: Claims, Head Moves, And Label Sweeps

Several writers act on the same PR set at once: the autonomous factory, parallel reviewer lanes,
and the owner. One writer per branch is the rule; the following make that rule observable.

- **Read live state immediately before every mutation** (`headRefOid`, labels, the last ~20 minutes
  of issue events and comments). If another writer pushed, relabeled, or commented on that PR inside
  the window and the activity is not yours, park the PR (`awaiting_reviewer` or `blocked_external`,
  reason `active_writer`) and move on. Do not race; do not retry on the same head.
- **Announce a claim before mutable work.** Post one bounded PR comment
  `review-claim: <lane-id> @ <head-sha> until <UTC>` (default 90 minutes) before pushing, editing
  the body, or changing labels; release it with `review-claim: released @ <head-sha>` on exit. An
  unexpired claim from another lane is an active writer. Until repository tooling enforces this
  marker it is advisory between reviewer lanes; the factory's own writes remain authoritative.
- **Content-identical head moves do not require a full re-review.** When the head advanced only by
  a `main` refresh (verify: `git diff origin/main...<old-head> -- <changed-files>` equals
  `git diff origin/main...<new-head> -- <changed-files>`), the prior findings transfer, but every
  exact-head carrier (`gate-verdict`, `base-policy`, `pr-metadata`) must be republished at the new
  head after hosted CI is green there. Any content change in the delta gets a focused review of the
  delta before republishing.
- **Label sweeps are authoritative.** If the factory or owner removes `merge-ready` (or applies
  `needs-review`, `deferred`, a hold label) without a comment, treat it as a live ruling: publish the
  exact-head evidence, report the PR as "one label away" with what is missing, and do not re-apply
  the label in the same run.
- **Successor / superset detection.** Before promoting a PR, list other open PRs on the same issue
  (`gh pr list --state open --search "<issue> in:body" --json number,files`) and compare changed
  files. If another open PR contains this PR's diff, record reciprocal `supersedes` /
  `superseded-by` lines in both bodies and choose explicitly: promote the reviewed subset and note
  that the successor must rebase to its residual, or hold the subset in favour of the successor.
  Never let both reach `merge-ready`.

## PR Label, Conversation-Comment, And Publication Helpers (REST-backed)

On affected GitHub CLI versions, both `gh pr edit <number> --add-label` and
`gh pr view <number> --comments` fail inside the GraphQL client with the retired
Projects Classic field (`repository.pullRequest.projectCards`) and, worse, can
exit `0` while emitting the error and no usable content (issue #6496). Neither
operation needs Projects Classic data, so perform them through the REST-only
helpers instead of the broad `gh pr` commands:

```bash
# merge-ready label add/remove (verify-on-write, pure REST issues-labels endpoint)
uv run python scripts/dev/gh_pr_label_rest.py add <number> \
    --label merge-ready --expected-head-sha <head_sha> --repo ll7/robot_sf_ll7
uv run python scripts/dev/gh_pr_label_rest.py remove <number> \
    --label merge-ready --repo ll7/robot_sf_ll7

# PR conversation comments, drop-in for `gh pr view <number> --comments`
# (pure REST repos/{repo}/issues/{n}/comments; no projectCards field queried)
uv run python scripts/dev/gh_pr_comments_rest.py <number> --repo ll7/robot_sf_ll7
uv run python scripts/dev/gh_pr_comments_rest.py <number> --repo ll7/robot_sf_ll7 --plain

# PR conversation comment publication (REST issues-comments endpoint)
scripts/dev/gh_comment.sh pr <number> --repo ll7/robot_sf_ll7 --body-file <path>
scripts/dev/gh_comment.sh pr --current --repo ll7/robot_sf_ll7 --body-file <path>

# Exact-head review publication; re-reads PR state/head while holding the local writer lock.
# The merge-ready carrier gate reads this COMMENTED review directly from the PR reviews API.
uv run python scripts/dev/gh_pr_review_rest.py <number> --event COMMENT \
    --body-file <path> --expected-head-sha <head_sha> --repo ll7/robot_sf_ll7
```

Use the label helper whenever the review loop applies, reapplies, or removes
`merge-ready` (including the remove-and-reapply gate refresh in step 8 below).
Use the comment helper to re-read the conversation thread after a fix push
before resolving review threads. Use `gh_comment.sh pr` for ordinary top-level
PR conversation comments; it resolves the target through local/REST state and
posts through `issues/{number}/comments`. The COMMENTED review event in step 8
is a separate review-verdict operation and must use the guarded review helper
above; it is the canonical carrier consumed by `pr_carrier_gate.py`. A matching
top-level issue comment remains a compatibility carrier for older API paths, but
is not required when the review endpoint is available. If the helper reports the machine-readable
`review_skipped_stale_state`, re-read the PR and stop the write path; classify a
merged PR as `merged_externally` / `no_action`. PR header/title fields that do not
involve comments can still use `gh pr view <number> --json ...`; only the
label-edit and `--comments` paths hit the deprecated field. The REST helpers
fail closed on auth, malformed, or truncated payloads.

## State Machine

Each PR is in one state:
- `queued`
- `under_review`
- `fixing`
- `awaiting_ci`
- `awaiting_reviewer`
- `blocked_external`
- `deferred_scope`
- `author_decision`
- `merge_ready`
- `closed_out`

`author_decision` is the parking state for a PR whose remaining blocker is an author-reserved
ruling (see `## Decision-Required Triage`). It carries the `decision-required` label and a posted
decision packet; the reviewer still keeps the branch mergeable (refreshed, CI green) so the ruling
can dispatch immediately. It maps to the terminal report state `AUTHOR_DECISION_REQUIRED`;
`blocked_external` maps to `BLOCKED_EXTERNAL`, `deferred_scope` to `DEFER_RECOMMENDED`, and
`merge_ready` to `MERGE_READY`.

Avoid loops:
- do not bounce `fixing` ↔ `awaiting_ci` without changes affecting proof.
- do not re-diagnose a failure already classified as `shared_main_blocked` on another PR in the
  same run; reuse the classification.

### Mapping policy output to states

`scripts/dev/pr_loop_policy.py` (see Review Workflow) classifies PRs with its own vocabulary.
Map each policy classification to a state in this skill so policy output drives state transitions
deterministically:

| `pr_loop_policy.py` classification | Recommended action | This skill's state |
| --- | --- | --- |
| `pending_ci` | `wait_ci` | `awaiting_ci` |
| `failed_ci` | `inspect_failed_ci` | `fixing` if the failure is a fixable regression on a writable branch; `awaiting_ci` with reason `shared_main_blocked` if the failing check reproduces on `origin/main` (see the shared-main baseline section); else `blocked_external` |
| `failed_validation` | `verify_artifacts` | `fixing` if the validation failure is actionable on a writable branch, else `blocked_external` |
| `missing_artifacts` | `verify_artifacts` | `under_review` |
| `stale_worktree` | `refresh_snapshot` | `under_review` (re-snapshot the advanced head before deciding) |
| `stale_merge_base` | `refresh_snapshot` or record the bounded ordinary selector | `under_review` until exact-head base policy proof is current |
| `blocked_preflight` | `no_action` | `blocked_external` until the blocking preflight condition is resolved |
| `unknown_review_threads` | `await_review_threads` | `awaiting_reviewer` until a thread-capable snapshot is available |
| `pending_gate_verdict` | `await_gate_verdict` | `awaiting_reviewer` until current exact-head gate evidence is present |
| `pending_pr_metadata` | `reconcile_pr_metadata` | `under_review` until the final title/body digest is reconciled and re-reviewed |
| `ready_to_merge` | `mark_ready_candidate` | `merge_ready` only after the full proof bar in `## Proof and Validation` closes; otherwise `under_review` |
| `no_action` | `no_action` | keep the current state (`awaiting_reviewer`, `blocked_external`, `deferred_scope`, or `closed_out`) |

`ready_to_merge` is a candidate signal, not a merge decision: the policy only checks CI and label
presence, so `merge_ready` still requires the intended-design and proof gates below.

## Shared-Main Baseline Before Per-PR Diagnosis

A red `origin/main` makes unrelated PRs look defective and gets re-diagnosed once per PR. Before the
first per-PR CI diagnosis of a run:

1. `git fetch origin` and record `MAIN_SHA=$(git rev-parse origin/main)`.
2. In a branch-attached worktree of `origin/main` (some inventory tests call
   `git symbolic-ref HEAD` and error on a detached HEAD), run the base-sensitive marker suite from
   `uv run python scripts/dev/base_sensitive_selector.py` plus any check that failed identically on
   two or more queued PRs.
3. If it is green, proceed normally.
4. If it is red: record `shared_main_blocked @ <MAIN_SHA>` with the failing test id in the ledger,
   classify every PR whose only failing check matches it as `awaiting_ci` (reason
   `shared_main_blocked`), do not push per-PR "fixes" for it, and route one bounded repair: reuse an
   existing repair PR if one references the failure, else file a `friction` issue labeled
   `dependency:blocks-others` and open the smallest repair. After the repair merges, PRs clear by an
   ordinary `main` refresh; re-snapshot instead of re-diagnosing.

Repeat step 2 only when `origin/main` has moved and a new identical failure appears on two PRs.

## Decision-Required Triage

A `decision-required` label is a claim that only the author can move the PR. Verify the claim from
the actual diff before treating it as terminal; unresolved but reducible uncertainty must not reach
the author.

Author-reserved (keep `decision-required`, park in `author_decision`):

- (a) locked prose, dissertation or publication claim-status, or evidence-admission changes —
  editing a claim ledger, marking evidence admitted or paper-grade, changing what a manuscript may
  conclude, assigning per-case result or confidence states to research lines;
- (b) preregistration authorization — declaring a study registered, frozen, or execution-authorized
  (a preregistration *packet* whose validator hard-requires `domain_approval.status: pending` and
  `execution_allowed: false` is not an authorization and is reviewable);
- (c) releases, tags, publication, repository settings or rulesets, secrets, credentials, protected
  data, destructive or irreversible operations;
- (d) orchestrator, gate, or lifecycle authority *expansion*. A change that only narrows what may
  be auto-approved, auto-merged, or reported (fail-closed direction, provable as "every prior fail
  reason still exists and at least one is added") is not an expansion and is reviewable; state the
  direction with `file:line` evidence in the review.

Reducible (remove `decision-required` after the normal review path): diagnostic-only tooling with
an explicit `evidence_boundary: diagnostic_only` or equivalent, tests, fail-closed guards,
validators and preflights, docs that record a ruling the author already made in the linked issue,
schema or contract code that admits nothing. Cite the prior ruling (issue comment, `domain-approved`
label, decision packet outcome) when one exists; implementing a recorded ruling is not a new
decision.

For an author-reserved PR: still refresh the base, fix PR-attributable CI, and publish exact-head
review evidence so the ruling can dispatch a merge without further work. Then, unless an equivalent
packet already exists, post one packet (≤ 25 lines, `### Decision packet`): the question in one
sentence; what is author-reserved and why, with file paths; recommendation and strongest
alternative; the automatic consequence of each ruling (merge, split, close, follow-up issue);
reopen condition; head SHA and validation evidence. Use `decision-cockpit` for packet construction
and ruling dispatch when available. Split when it removes the residual decision: e.g. merge the
inert tooling half now and leave only the claim-bearing artifact for the author.

## Review Workflow

1. Build the broad review queue snapshot with
   `uv run python -m scripts.dev.snapshot_pr_queue --active` (labels, draft status, checks, head
   SHA, last update time) before broad `gh pr view` fields. For a review already scoped to one PR,
   use `uv run python -m scripts.dev.snapshot_pr_queue <pr-number>` instead.
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
   - require artifact-first delegated review from an orchestrator artifact directory outside every
     git worktree and validate in order: `result.json`, `RESULT.md`, `diffstat.txt`, and
     `validation.json`; inspect route evidence first, then run targeted local checks before raw logs;
     `RESULT.md` and `REVIEW.json` must never be staged or committed,
   - cap parent-thread raw output at about 200 lines; use `rg -l`, `rg --files`, bounded `sed -n`,
     and private artifacts instead of broad `rg -n .` or full file reads,
   - classify findings as fixable now, deferred, or blocker.

Before choosing the next action for any PR, consult the compact snapshot. For ordinary review-loop
triage, apply the machine-checkable state policy:

```bash
uv run python -m scripts.dev.pr_loop_policy --snapshot <queue-snapshot.json> --json
```

The policy classifies each PR into `pending_ci`, `failed_ci`, `failed_validation`,
`missing_artifacts`, `stale_worktree`, `stale_merge_base`, `blocked_preflight`,
`unknown_review_threads`, `pending_gate_verdict`, `pending_pr_metadata`, `ready_to_merge`,
or `no_action` and recommends one bounded action under the loop budget. Use the policy decision
to avoid ad-hoc state inspection.

For PR babysitting or handoff, prefer the conservative one-shot babysitter snapshot:

```bash
uv run python scripts/dev/pr_babysitter_snapshot.py <pr-number> --expected-head-sha <sha> --json
```

Use it when the next question is whether to wait, diagnose failed CI, process review feedback, stop
because the PR closed or went stale, or perform a final merge-readiness review. The babysitter
helper is route evidence only and does not perform GitHub-visible writes.

4. Fix actionable items on writable branches; commit and push. After every fix or revision push,
   rebuild the final title/body from the resulting diff, claims, validation, and follow-ups and run
   `uv run python scripts/dev/gh_pr_body_rest.py --reconcile`. Treat an unchanged result as a valid no-op; a
   changed title is justified only when scope, intent, type, or issue linkage changed.
   The metadata digest binds body *bytes*, not body *truth*, so also check the narrative before
   any label write: every 40-hex SHA in the body must resolve (`git cat-file -e <sha>^{commit}`
   after `git fetch origin`) and every exact-head carrier must equal the live `headRefOid` — never
   complete a SHA from a prefix; and a body that still says "not merge-ready", "remains
   unapproved", "pending independent review", or "do not merge" must be re-narrated before
   `merge-ready` is applied, otherwise that sentence is what gets squashed into `main`
   (issues #7448, #7491).
   Review-only worktrees are not writable publication lanes: create them with
   `scripts/dev/create_worktree.sh --mode review`, and use the guarded
   `review_worktree_guard.py integrate` command for any synthetic merge. Do not push an explicit
   refspec from a review worktree.
5. Validate per required tier, including the PR title/body contract after reconciliation.
   For any PR whose declared base is older than current `main`, run
   `uv run python scripts/dev/check_base_sensitive_gates.py --pr <number> --json` against the exact
   head. If the selector is `base_sensitive`, refresh onto current `main`, rerun the focused subset,
   and keep the normal stale-base hold until that proof is current. If the selector is `ordinary`,
   include `base-policy: ordinary-cas @ <head-sha>` in the trusted exact-head review evidence; the
   guarded merger will perform the immediate current-main CAS before merging. Missing or unknown
   selector evidence remains blocked.
6. Re-query unresolved review threads after push, metadata reconciliation, and verification before resolving anything, especially
   when moving draft PRs to ready or when bot reviewers were previously pending or skipped.

For an explicitly stacked PR set, capture `scripts/dev/stacked_prs.py status --prs <root> <tip>
--json` before applying review labels. Use its root-to-tip base alignment and exact-head fields as
the review snapshot; use `retarget` or `sync` only from a clean worktree and only with the expected
heads recorded from that snapshot. After any base retarget, push, merge, or GitHub automatic base
advance, rerun the affected PR's focused validation, metadata reconciliation, review evidence, and
thread snapshot. A child PR is not merge-ready merely because its parent merged: the stacked helper
stops after advancing the child until fresh CI and exact-head evidence are current.

7. Resolve review threads only after the post-push thread snapshot confirms the fixes still cover all
   actionable comments.
8. After the full proof bar closes, reconcile the final title/body one more time and compute its
   exact metadata digest. Immediately before the write, use
   `scripts/dev/gh_pr_review_rest.py` with the captured full head SHA to re-read PR lifecycle
   state and head, then post an exact-head review-evidence comment by submitting a GitHub
   COMMENTED review naming the reviewed SHA, the validation, findings disposition, any
   single-account waiver, and
   `pr-metadata: reconciled @ <digest>` alongside `gate-verdict: accepted @ <head_sha>`. Then update
   `merge-ready` through `gh_pr_label_rest.py` with the same expected head SHA. Both writes
   return `review_skipped_stale_state` without mutating a PR if it is no longer open or its
   head moved. The
   review event refreshes the source-head queue gate after the verdict. If review submission is
   unavailable, a matching top-level compatibility carrier may be used only when the guarded label
   gate can read it; otherwise record the gate-refresh blocker.

   Start the review body with `## Exact-head self-review` (canonical) or
   `## Exact-head implementation review`. The guarded carrier parser requires one of these explicit
   headings plus the live full head SHA; generic review prose does not satisfy the merge-ready gate.
9. When CI is the only remaining external gate, put the PR in `awaiting_ci` and use compact,
   bounded one-shot polling in non-TTY agent sessions instead of `gh pr checks --watch`:
   `uv run python scripts/dev/watch_pr_ci_status.py <number> --once --json --expected-head-sha <sha>`.
   Inspect JSON/job state first with `gh run view <run-id> --json status,conclusion,jobs` or the
   repo CI helpers. Fetch raw logs only for the relevant failed or completed job, return bounded
   excerpts second with grep/tail, and explicitly label those snippets as bounded excerpts. Keep full logs in private artifacts. Avoid fetching
   `body,comments,reviews,files,statusCheckRollup` together unless the review task explicitly needs
   that full surface. Use `.agents/skills/goal-autopilot/SKILL.md` "Async CI Wait Policy" instead of
   idling the review loop when other safe PR or cycle work remains.
   Under hosted-runner starvation (checks queued with zero elapsed time for longer than one bounded
   poll budget), stop waiting: publish the exact-head evidence you already have, state in the PR
   comment exactly what remains ("apply `merge-ready` once checks conclude green at `<head>`"),
   and exit. Do not leave background pollers or sleep loops running after handoff.
10. Update the active ledger before any CI wait or final handoff. Route completion is not task
   completion until the main agent has verified proof, GitHub state, and cleanup.

## Shared Resource Budget

Reviewer lanes share one GitHub token, one hosted-runner pool, and one host. Under parallel review:

- Check `gh api rate_limit` before a polling cycle; when `core.remaining < 300` (or GraphQL is
  exhausted), stop polling, do local work, and resume after the reset time. Cycle bounded polls
  across PRs rather than spinning on one; REST helpers count against the same budget.
- Every push and body edit re-triggers hosted checks. Batch fixes into one push, reconcile the body
  once after the final push, and do not push a `main` refresh only to move a queued check.
- Create worktrees on repository disk (`.worktrees/<name>`) or check free space first (`df -h` on
  the target filesystem); tmpfs such as `/dev/shm` fills under concurrent `uv sync`. Reuse the main
  checkout's environment where the repo helper allows it, remove your worktrees on exit, and never
  delete another lane's worktree unless it is provably abandoned (no writer, stale for hours).
- Detached-HEAD worktrees make some inventory tests fail (`git symbolic-ref HEAD` exits 1); use a
  branch-attached worktree before attributing such a failure to a PR.

## Intended Design And Follow-Up Gate

Before applying `merge-ready`, reviewers must explicitly answer:

- What was the intended design or issue contract?
- Does the implementation behavior match that intent, including tests, docs, and PR claims?
- If the PR intentionally narrowed scope, is the narrowed scope named in the PR or issue and still
  useful on its own?
- Are remaining gaps current-PR blockers, bounded follow-up issues, or handoff-only notes?

For an intentionally partial PR, require `Refs #<parent>` rather than `Closes`/`Fixes`, keep the
parent issue open, and create and cross-link a successor issue that owns every residual acceptance
criterion and its proof tier. The reviewer is authorized to make these PR-body, comment, label, and
successor-issue writes. Withhold `merge-ready` if narrowing would strand work required for a public
claim, benchmark interpretation, metric/schema correctness, or safe runtime behavior.

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

## Single-Account Internal-Review Waiver

When the repository is operated through one effective GitHub account and no independent review
identity exists, waive the distinct-account approval requirement. Record an internal review comment
that names the exact reviewed head SHA, commands and artifacts checked, findings disposition, and
the single-account waiver reason. Exact-head evidence remains mandatory: any head change invalidates
the review record and `merge-ready` decision until the new head is reviewed and validated.

The waiver does not bypass branch protection, an explicitly requested external reviewer, unresolved
actionable threads, CI, or any domain-aware approval required for evidence classification or
paper-facing claims.

## Proof and Validation

Apply minimum tier by change surface:

- Tier 0: documentation and formatting scope with targeted checks and lint.
- Tier 1: integration and replay changes, CLI runtime smoke, PR readiness.
- Tier 2: planner, metric, scenario, and benchmark behavior.
- Tier 3: campaign-level statistical claims or paper-facing evidence.

`merge-ready` conditions:
- linked issue contract and intended design satisfied, or intentionally narrowed with explicit
  rationale, `Refs #<parent>`, an open parent, and linked successor issues,
- scope matches contract and tests and CI proof are current for reviewed SHA,
- stale-base handling is explicit: current-base subset proof for `base_sensitive` changes, or
  trusted exact-head `ordinary-cas` evidence for the final current-main compare-and-swap path,
- unresolved actionable review threads closed via GitHub review-thread resolution, with any
  single-account waiver recorded for the exact reviewed SHA,
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
- Keep review artifacts outside git worktrees; never commit `RESULT.md` or `REVIEW.json`.
- Parallel reviews may use isolated read-only or fix worktrees when the parent orchestrator assigns
  separate PRs. Never let two reviewers mutate the same branch, and refresh the exact head before
  applying labels or resolving threads. Apply `## Concurrent Writers` (claim marker, active-writer
  window, content-identical head moves, label sweeps) whenever more than one writer is live.

## Output Requirements

For each reviewed PR, report:
- PR number, head SHA, queue state transitions, and one terminal state
  (`MERGE_READY`, `AUTHOR_DECISION_REQUIRED`, `BLOCKED_EXTERNAL`, `DEFER_RECOMMENDED`, or
  `MERGED` / `CLOSED_SUPERSEDED` when that already happened),
- validation tier and executed commands,
- fix commits,
- `merge-ready` decision + confidence, or the exact remaining step when evidence is published but
  the label is withheld ("one label away": which check, which writer, which ruling),
- for `author_decision`, the packet location and the recommendation,
- blockers and `follow-up` issues,
- parked / racing PRs with the active writer named,
- artifact classification decision and worktree cleanup status.

## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.
See `## Trigger Boundary` for the precise in-scope and out-of-scope actions.

## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.
