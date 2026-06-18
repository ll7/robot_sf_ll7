# Issue #2909 GitHub Payload Overfetch Audit 2026-06-17

Status: Current

Issue: [#2909](https://github.com/ll7/robot_sf_ll7/issues/2909)

Date: 2026-06-17

## Summary

The workflow GitHub helpers in `scripts/dev/` are already mostly field-bounded. The remaining
larger payloads are either intentionally opt-in raw artifacts, bounded excerpt snapshots, or
required by the compact routing output they produce.

The main residual cost is not a single broad `--json` field set. It is repeated per-issue claim
state checks in issue-list snapshots, where each listed issue calls `git ls-remote` through
`scripts/dev/issue_claim.py`. That cost is local/Git remote traffic rather than GitHub JSON payload
overfetching. It is a reasonable follow-up if issue queue snapshots become slow at larger limits.

## Audit Table

| Script | GitHub surface | Fetched fields or payload | Used for | Classification | Recommendation |
| --- | --- | --- | --- | --- | --- |
| `scripts/dev/check_pr_ci_status.py` | `gh pr view <pr> --json ...` | `number`, `title`, `state`, `mergeable`, `headRefName`, `headRefOid`, `statusCheckRollup`, `reviews` | CI rollup, SHA guard metadata, mergeability, branch/head display, and review-state counts | Required | Keep field set. It is already narrower than raw PR data and all fields feed output. |
| `scripts/dev/snapshot_pr_queue.py` | `gh pr view` / `gh pr list --json ...` | `number`, `title`, `state`, `isDraft`, `labels`, `url`, `headRefName`, `headRefOid`, `mergeable`, `statusCheckRollup`, `reviews`, `comments` | Active PR triage, review summaries, comment excerpts, labels, CI status, stale-head checks | Required but body-bearing | Keep field set for current snapshot contract. It fetches review/comment bodies but truncates them before stdout. |
| `scripts/dev/snapshot_pr_queue.py` | GraphQL `reviewThreads(first:12)` | Thread id/resolution/path/line and first two comment bodies per thread | Optional `--review-threads` unresolved-thread snapshot | Intentional bounded read | Keep. It omits raw diff hunks and full bodies from stdout. |
| `scripts/dev/snapshot_pr_queue.py` | REST `pulls/<n>/comments` | Raw review comments including diff hunks | Optional `--raw-review-comments-artifact` | Intentional raw artifact | Keep behind explicit artifact path. It is never printed to stdout by default. |
| `scripts/dev/snapshot_issue_batch.py` | `gh issue view <n> --json ...` | `number`, `title`, `body`, `state`, `labels`, `url`, `assignees` | Explicit issue snapshots with bounded body excerpts and claim/classification metadata | Required | Keep. Body is needed for explicit issue context and is truncated before stdout. |
| `scripts/dev/snapshot_issue_batch.py` | `gh issue list --json ...` | `number`, `title`, `state`, `labels`, `url`, `assignees` | Claimable, blocked-external, and active-portfolio snapshots | Required | Keep. No body field is fetched in list modes. |
| `scripts/dev/check_pr_followups.py` | Optional `gh issue view` through follow-up verification | Linked follow-up issue state | Verifies linked follow-ups are open when explicitly requested | Required on opt-in path | Keep. The command is only used for the stricter verification mode. |
| `scripts/dev/gh_comment.sh` | `gh pr view --json number --jq .number` | Current PR number only | Resolves `--current` target for comment helper | Required | Keep. This is already scalar. |
| `scripts/dev/publication_scout_linter.py` | Local JSON payload input for GraphQL publication result | `errors` payload from a preexisting result file | Deterministic classifier for publication failures | Not a live fetch | No change. |
| `scripts/dev/watch_pr_ci_status.py` | Example command uses `gh pr view --json headRefOid` | Head SHA only | Documentation/example for SHA guard | Required example | Keep. |

## Residual Cost Classes

- `snapshot_issue_batch.py` calls `status_issue()` once per listed issue. This is useful for claim
  safety, but it can dominate large issue snapshots because it repeats `git ls-remote` work.
- `snapshot_pr_queue.py` intentionally fetches `reviews` and `comments` for headline PR snapshots
  so parent agents do not open broader PR payloads or raw review bodies later.
- Raw review comments with diff hunks are opt-in and stored in a caller-named artifact path; this is
  the right boundary for rare review-comment repair workflows.

## Follow-Up Options

Follow-up issue [#2986](https://github.com/ll7/robot_sf_ll7/issues/2986) tracks the next smallest
improvement: a claim-status batch cache for `snapshot_issue_batch.py`. That change should fetch
`refs/heads/agent-claims/*` once per snapshot, then classify issue claims from that map. Acceptance
should include parity tests against the current per-issue `status_issue()` behavior.

No immediate field-reduction patch is recommended from this audit. Cutting any current PR or issue
fields would remove routing evidence that the compact snapshots are designed to provide.

## Validation

- Inspected GitHub call sites in `scripts/dev/check_pr_ci_status.py`,
  `scripts/dev/snapshot_pr_queue.py`, `scripts/dev/snapshot_issue_batch.py`,
  `scripts/dev/check_pr_followups.py`, `scripts/dev/gh_comment.sh`,
  `scripts/dev/publication_scout_linter.py`, and `scripts/dev/watch_pr_ci_status.py`.
- Verified the audit note and index paths locally with `test -f`.
