---
name: gh-pr-merger
description: Guarded PR merger; merges merge-ready PRs after verifying label, CI status, branch
  protection, and preflight checks.
category: github-pr
kind: atomic
phase: verification
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
aliases:
- pr-merger
- guarded-pr-merge
---

# GH PR Merger

Use this skill when a PR is `merge-ready` and the owner or parent orchestrator has authorized a
bounded guarded-merge run.

This skill is intentionally restricted: it never force-pushes, never rewrites
history, and stops on any auth/permission/CI failure.

## Merge Queue Gate Parity (issue #6274)

The fail-closed preflight below governs guarded merges this skill performs. The
same preflight is enforced **queue-side** by the `Merge Queue Gate` required
status check (`.github/workflows/merge-queue-gate.yml`, backed by
`scripts/dev/merge_queue_gate.py`), so the GitHub native merge queue and any
external/parallel auto-merge dispatcher that routes through it cannot bypass
`merge-ready`, the exact-head `gate-verdict: accepted` trailer, the exact final
PR metadata digest, unresolved
threads, or an explicitly requested reviewer. Comment or review verdict trailers count only when
GitHub reports the author as a repository owner, member, or collaborator; an untrusted contributor
cannot self-approve a new head under a retained label. The queue gate also fails closed
unless the live queue uses GitHub's `ALLGREEN` ("Only merge non-failing pull
requests") strategy, which prevents a
passing tail entry from carrying an earlier ungated entry through a grouped
merge. It also fails closed when a trusted unexpired exact-head `review-claim` marks the review
worker as active; the claim must be released after exact-head evidence is published and before
`merge-ready` admission. That workflow is the merge-queue entry point for this contract; this
skill remains the binding authority for guarded merges it executes directly.
See `docs/dev_guide.md` ("Merge queue gate") for the required-check toggle and
the audit record shape (`merge_queue_gate.v1`).

## Single-account merge receipt (issue #7669)

Every mutating merge path consumes the versioned
`single_account_merge_receipt.v1` produced and verified by
`scripts/dev/single_account_merge_receipt.py`. The receipt binds the exact PR head and
metadata digest to terminal required checks, independent implementation-review evidence,
resolved threads, requested-reviewer state, and separate dependency, draft, domain,
scientific/evidence, legal/release, security, and merge holds. A single-account waiver may
only document the absence of a distinct human implementation reviewer; it cannot clear any
other hold or replace hosted checks, domain approval, evidence, legal, release, security, or
dependency authority. The receipt owner performs the final expected-head compare-and-swap
merge and records GitHub's returned merge SHA. When an ordinary PR's recorded base predates
current `main`, the receipt additionally binds the complete changed-file record inventory
(filename, status, and prior rename path), exact
base/head/current-main content refs for every changed Python test marker candidate (including a
renamed file's prior path), a trusted full-40-hex exact-head `base-policy: ordinary-cas` carrier,
and immediate current-main/head CAS. GitHub's 3,000-file response cap cannot prove completeness and
therefore fails closed.
This qualifies only the gate's `stale_merge_base` reason; every additional gate reason remains
blocking. Report and validate modes are read-only.

If the merge PUT returns a transport or server error and the immediate PR readback is not
terminal, the receipt records a bounded `post_error_reconciliation` object with the PR readback
and one read-only current-`main` ref readback. Treat that result as `unconfirmed`: preserve the
original error, do not infer success from the `main` ref, and do not retry without a new external
state change.

The authority fixture at
`scripts/dev/single_account_merge_authority_fixture.v1.json` enumerates the allowed callers
and is checked in tests. Do not add a second direct merge endpoint or a raw merge command to
this skill; update the fixture and receipt owner together if the repository adds a legitimate
caller.

## Trigger Boundary

Use this skill when the user asks to merge approved, `merge-ready` PRs.

Do not use it for:
- merging without the `merge-ready` label,
- force-pushing or branch rewriting,
- merging draft PRs,
- merging PRs with failing CI,
- merging PRs targeting non-`main` branches without explicit override.

## Read First

- `AGENTS.md`
- `.agents/skills/goal-pr-review/SKILL.md`
- `docs/code_review.md`
- `docs/dev_guide.md` ("Merge queue gate", issue #6274)
- `.github/workflows/merge-queue-gate.yml`
- `scripts/dev/merge_queue_gate.py`
- `scripts/dev/single_account_merge_receipt.py`
- `scripts/dev/single_account_merge_receipt.v1.schema.json`
- `scripts/dev/single_account_merge_authority_fixture.v1.json`
- `scripts/dev/base_sensitive_selector.py`
- `scripts/dev/check_base_sensitive_gates.py`
- `scripts/dev/check_pr_current_base_cas.py`
- `.github/PULL_REQUEST_TEMPLATE/pr_default.md`

## Preflight

Run the declared runtime preflight before processing a batch:

```bash
uv run python scripts/dev/check_skills.py --preflight gh-pr-merger
```

Before each merge operation, verify:

1. Current GitHub state has the `merge-ready` label. Labels are the source of truth; do not infer
   merge authorization from Projects, dashboards, a local ledger, or a worker report. If absent,
   skip and report.
2. PR is not a draft. If draft, skip and report.
3. PR targets `main` (or the explicitly allowed base branch).
4. Read the live PR title/body through the REST-backed metadata path, compute the exact metadata
   digest, and verify a trusted `pr-metadata: reconciled @ <digest>` trailer matches it. If the
   trailer is missing or stale, skip and report; the merger verifies metadata but never invents or
   mutates the final narrative.
5. CI checks are passing (use `uv run python scripts/dev/check_pr_ci_status.py <number>`).
   In non-TTY agent sessions, prefer bounded polling over `gh pr checks --watch`:
   `uv run python scripts/dev/check_pr_ci_status.py <number> --poll-attempts 20 --poll-interval 30`.
   Exit code `2` means checks were still queued or in progress after the polling budget; inspect the
   listed check URLs or run `gh run view <run-id> --json status,conclusion,jobs` for job state.
6. No merge conflicts exist (`gh pr view <number> --json mergeable`).
7. **Risk-tiered base policy and final CAS** (issues #5559/#6272): first run
   `uv run python scripts/dev/check_base_sensitive_gates.py --pr <number> --json`.
   The changed-file selector must return `base_sensitive` or `ordinary`; `unknown` or missing
   inventory fails closed. For `base_sensitive`, run the existing
   `python scripts/dev/check_pr_merge_staleness.py <number>` and
   `uv run python scripts/dev/check_base_sensitive_gates.py --pr <number> --run-subset --json`.
   Any stale, unknown, or failed result skips the PR; the author must update the branch and rerun
   CI. For `ordinary`, a trusted exact-head review must carry
   `base-policy: ordinary-cas @ <head-sha>` and the merger must not infer that status from a local
   snapshot. The guarded receipt owner retains every normalized changed-file record, derives the
   exact test candidate set from those records, and classifies marker content at immutable PR
   base/head/current-main refs, includes both paths of renamed candidates, verifies the current-main
   commit ref before treating a missing path as absent, preserves the full carrier, and runs the
   equivalent immediate `check_pr_current_base_cas.py` proof against the expected head and current
   `main`; inspect its `ordinary_cas` field before apply. A changed-file result at GitHub's
   3,000-file cap is incomplete and fails closed.
   A non-passing CAS result, moved `main`, moved head, missing provenance, or non-`main` target
   skips the PR. The final merge still uses `--match-head-commit "$HEAD_SHA"`. See issue #6272.
8. The PR has no unresolved actionable review threads or outstanding explicitly requested external
   reviewers. A distinct-account approval may be waived only under `goal-pr-review`'s documented
   single-account internal-review waiver.
9. Branch protection rules on `main` allow merges from the current actor.
10. The current head SHA exactly matches the SHA named in the `merge-ready` review evidence. A
   single-account waiver never waives exact-head evidence; any head change requires re-review.

If any preflight check fails, report the specific failure and do not merge.
Do not retry preflight on the same PR without a state change.

## Autonomous Merge Authority And Deletion Boundary

Starting this bounded merge run authorizes the merger to update merge-status labels, post preflight
or outcome comments, and execute the guarded squash merge without another per-PR confirmation. The
merger may delete the merged feature branch only after verifying it contains no unique unpreserved
work. It must not force-push, rewrite history, resolve substantive review findings itself, or bypass
branch protection.

Owner approval is required before deleting durable scientific artifacts, experiment records or
runs, or GitHub releases. A merge command, cleanup option, or stale local cache never implies that
approval. Store merger/review control-plane artifacts outside git worktrees, and never commit
`RESULT.md` or `REVIEW.json`.

## Label, Comment Read, And Publication Helpers (REST-backed)

On affected GitHub CLI versions, `gh pr edit <number> --add-label` and
`gh pr view <number> --comments` fail inside the GraphQL client with the retired
Projects Classic field (`repository.pullRequest.projectCards`) and can even
exit `0` while emitting the error and no usable content (issue #6496). Neither
operation needs Projects Classic data. When this merger updates labels or reads
conversation comments, use the REST-only helpers instead of the broad `gh pr`
commands:

```bash
# add/remove a label (verify-on-write, pure REST issues-labels endpoint)
uv run python scripts/dev/gh_pr_label_rest.py add <number> \
    --label <label> --repo ll7/robot_sf_ll7
uv run python scripts/dev/gh_pr_label_rest.py remove <number> \
    --label <label> --repo ll7/robot_sf_ll7

# PR conversation comments, drop-in for `gh pr view <number> --comments`
# (pure REST repos/{repo}/issues/{n}/comments; no projectCards field queried)
uv run python scripts/dev/gh_pr_comments_rest.py <number> --repo ll7/robot_sf_ll7

# publish a PR conversation comment (REST; body file avoids shell quoting)
scripts/dev/gh_comment.sh pr <number> --repo ll7/robot_sf_ll7 --body-file <path>
```

This skill verifies the `merge-ready` label rather than applying it, but any
post-merge status-label cleanup or preflight/outcome comment context that needs
the conversation thread must go through these REST paths. Publish ordinary PR
conversation comments with `gh_comment.sh pr`, which resolves the target through
local/REST state and posts to the REST issues-comments endpoint. Read-only PR
header fields still use `gh pr view <number> --json ...`; only the label-edit
and `--comments` paths hit the deprecated field. The REST helpers fail closed on
auth, malformed, or truncated payloads.

## Merge Workflow

For a root-to-tip stack, `scripts/dev/stacked_prs.py merge-cascade` is the guarded coordinator for
the sequential mechanics. It requires `--apply` plus an exact `--expected-head PR=SHA` for every
stack entry, performs only a squash merge, verifies the remote merged state, and checks whether
GitHub automatically retargeted the next PR to `main`. If the next base changed or needs explicit
retargeting, it stops with a fresh-CI handoff; rerun the full merger preflight after CI and exact-head
review evidence are current. Never use it to bypass this skill's `merge-ready`, metadata, thread,
requested-reviewer, base-policy, or branch-protection checks, and never run stack mutations
concurrently against the same PR or branch.

1. List open PRs with `merge-ready` label:
   ```bash
   gh pr list --state open --label merge-ready --json number,title,headRefName,baseRefName,mergeable,statusCheckRollup
   ```
2. For each PR:
   - Run preflight checks.
   - Update the active delegation ledger from `.agents/skills/goal-autopilot/SKILL.md` with the PR
     number, head SHA, preflight status, merge command status, cleanup status, and next action.
   - If all pass, build and validate the exact-head receipt before the compare-and-swap merge:
     ```bash
     uv run python scripts/dev/single_account_merge_receipt.py \
       --repo "$OWNER/$REPO" --pr <number> --mode report-only \
       --output /tmp/pr-<number>-single-account-receipt.json
     uv run python scripts/dev/single_account_merge_receipt.py \
       --repo "$OWNER/$REPO" --pr <number> --mode apply \
       --receipt-file /tmp/pr-<number>-single-account-receipt.json
     ```
     The report must be `ready`; validate mode rereads live evidence and apply mode performs
     the one expected-head squash merge through the receipt owner. If any lookup or receipt
     check fails, stop before attempting the merge. For a root-to-tip stack, use
     `scripts/dev/stacked_prs.py merge-cascade --apply`; it builds the same receipt and
     delegates the merge operation to the receipt owner.
   - If merge fails, run diagnostics and handle the failure (see Merge
     Command Failure below) before continuing to the next PR.
   - Update the active ledger after remote merge verification, branch deletion, claim release, and
     worktree/artifact cleanup decisions. Keep remote merge success separate from local cleanup
     failures.
3. Report merged PRs, skipped PRs with reasons, and any failures.

Do not merge multiple PRs in parallel. Process sequentially.

## Delegation Failure Recovery

- Merge conflict:
  - Do not attempt to resolve conflicts. Report the conflict state and leave the
    PR open.
  - The author or `goal-pr-review` must fix the branch before retry.

- CI check failure:
  - Do not merge. Report the failing check name and URL.
  - Do not override CI failure unless the user explicitly requests override, and
    only after recording the override rationale.
  - If checks are queued or in progress, do not use unbounded watch mode in non-TTY sessions.
    Poll with `scripts/dev/check_pr_ci_status.py --poll-attempts ... --poll-interval ...`; logs
    may be unavailable until a job completes, so use `gh run view <run-id> --json status,conclusion,jobs`
    to distinguish queued, in-progress, and completed states before fetching logs.

- Auth/permission failure:
  - Stop immediately. Report the failing command, exit code, and stderr.
  - Do not retry without fixing the credential or permission gap.

- Branch protection rejection:
  - Record the rejection reason from the `gh` CLI output.
  - Report the specific protection rule that blocked the merge.
  - Do not attempt to bypass branch protection.

- Merge command failure:
  - Run diagnostics to distinguish remote-side success from true failure:
    ```bash
    gh pr view <number> --json state,mergedAt,mergeCommit,headRefOid,headRefName
    ```
  - If `state == "MERGED"` and `mergedAt` is set, the remote merge succeeded
    but local cleanup (branch deletion, cache update) may have failed. Report
    the merged commit SHA and note the cleanup failure separately. Do not
    retry; the PR is already merged.
  - If `state` is not `"MERGED"`, the merge genuinely failed. Record the
    `headRefOid`, exit code, and stderr. Do not retry without an external
    state change.
  - Never retry a PR whose remote state is already `"MERGED"`.

## Anti-Loop Rules

- Do not retry merging the same PR if preflight or merge command failed without
  an external state change (new CI run completed, conflicts resolved, label added).
- Never retry a PR whose remote state is already `"MERGED"`, even if the local
  receipt-apply command exited nonzero. Check with
  `gh pr view <number> --json state` before any retry attempt.
- After two sequential PRs fail the same preflight check, stop and report the
  pattern instead of continuing.

## Race-Condition / Multi-Agent Safety

- Before merge, verify the PR head SHA has not changed since the last review.
  If changed, skip and report that the PR needs re-review.
- Do not merge a PR that has unresolved actionable review threads.
- Honor explicitly requested external reviewers. Lack of a second internal account is not a blocker
  only when the exact-head single-account waiver is recorded.
- Multiple machines may prepare or review isolated PRs in parallel, but this merger must process
  merges sequentially and re-read labels, base, checks, threads, and head SHA immediately before
  each merge.

## Confidence

- `High`: PR merged successfully with all preflight checks passed.
- `Medium`: PR skipped because preflight failed with a clear, fixable reason.
- `Low`: auth/permission/environment blocker.

## Required Output

For each attempted merge, report:
- PR number and head SHA,
- preflight check results (pass/fail per check),
- merge command and exit code,
- merged URL or skip reason,
- branch deletion status,
- any delegation failure and recovery action.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.


## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.


## Output

Return the schema named by the `output_schema` frontmatter field, or a compact equivalent when the caller does not require YAML.
