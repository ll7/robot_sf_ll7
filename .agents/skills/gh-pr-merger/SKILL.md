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
4. CI checks are passing (use `uv run python scripts/dev/check_pr_ci_status.py <number>`).
   In non-TTY agent sessions, prefer bounded polling over `gh pr checks --watch`:
   `uv run python scripts/dev/check_pr_ci_status.py <number> --poll-attempts 20 --poll-interval 30`.
   Exit code `2` means checks were still queued or in progress after the polling budget; inspect the
   listed check URLs or run `gh run view <run-id> --json status,conclusion,jobs` for job state.
5. No merge conflicts exist (`gh pr view <number> --json mergeable`).
6. **Merge-staleness check**: run `python scripts/dev/check_pr_merge_staleness.py <number>`.
   Exit code `1` means main has moved since the PR's CI ran; skip and report the
   PR as stale — the author must update the branch and re-run CI before merge.
   Exit code `2` means the check could not determine staleness (API error, no
   workflow-run metadata); log a warning and continue with remaining preflight
   checks.  See issue #5389 for context.
7. The PR has no unresolved actionable review threads or outstanding explicitly requested external
   reviewers. A distinct-account approval may be waived only under `goal-pr-review`'s documented
   single-account internal-review waiver.
8. Branch protection rules on `main` allow merges from the current actor.
9. The current head SHA exactly matches the SHA named in the `merge-ready` review evidence. A
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

## Merge Workflow

1. List open PRs with `merge-ready` label:
   ```bash
   gh pr list --state open --label merge-ready --json number,title,headRefName,baseRefName,mergeable,statusCheckRollup
   ```
2. For each PR:
   - Run preflight checks.
   - Update the active delegation ledger from `.agents/skills/goal-autopilot/SKILL.md` with the PR
     number, head SHA, preflight status, merge command status, cleanup status, and next action.
   - If all pass, record the PR head SHA and merge using squash merge:
     ```bash
     HEAD_SHA=$(gh pr view <number> --json headRefOid --jq .headRefOid)
     gh pr merge <number> --squash --delete-branch --match-head-commit "$HEAD_SHA"
     ```
     If `HEAD_SHA` is empty or the `gh pr view` command fails, stop and report
     the lookup failure before attempting the merge.
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
  `gh pr merge` command exited nonzero. Check with
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
