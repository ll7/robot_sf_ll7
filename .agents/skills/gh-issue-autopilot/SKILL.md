---
name: gh-issue-autopilot
description: Autonomous issue-to-PR workflow from next eligible issue to ready PR with consistent metadata
  handling.
category: github-issue
kind: orchestrator
phase: implementation
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to:
- gh-issue-sequencer
- implementation-verification
- pr-ready-check
- gh-pr-opener
- artifact-provenance
output_schema: issue_to_pr_summary.v1
aliases:
- issue-to-pr
- gh-issue-to-pr
---

# GH Issue Autopilot

Use this when the user asks to "take the next issue" and execute through to a ready PR.

Primary integration is with repo project state and PR opening; details of branch validation and detailed
issue creation are handled by child skills.

## Constants

- Repository: `ll7/robot_sf_ll7`
- Project: `ll7` Project `#5`
- Base branch: `main`
- Readiness baseline: `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`
- Workflow note: `docs/context/issue_713_batch_first_issue_workflow.md`

## Selection Policy

Choose the next issue in order:
1. Project status `Ready`
2. Project status `Todo`
3. Project status `Tracked`
4. Explicit user-requested issue

Within the same status, use `gh-issue-sequencer` as the source of truth for Project #5 ordering.
Tie-breakers after sequencing: no blocker labels, no linked PR, older open issue first, stronger evidence first.

Before accepting a ready candidate, inspect the issue body and recent comments for `source_pr`,
linked PRs, `Closes`/`Refs`, and explicit prerequisite PRs. Verify the referenced implementation
surface exists on current `origin/main` when the branch must start from clean main. If an open PR
already covers the issue, classify it as `covered_by_pr`, route it to `state:running`, and stop
instead of reimplementing. If the issue depends on an unmerged source PR, mark it blocked or
unavailable for clean-main work with the unblock condition "source PR merged to origin/main",
unless the user explicitly chooses a stacked-PR route.

### Exact merged-fix stale-evidence guard

Before auto-admitting a ready candidate, revalidate it against current `origin/main` history using
the issue's named symbol, failing test, and error signature — not issue-number references alone.
Search recently merged PR titles and bodies, and when a fix is named only by issue number, map its
merge commits back to the covering PR through the commit-to-pulls API so a fix filed under a
different issue number is still found. Then verify the candidate's failure signature, named symbol,
and failing file/line against current `origin/main` history and code. Classify the issue as
`covered_by_pr` and stop before claim or branch only when an exact merged fix implements the stated
boundary and its regression proof covers the reported failure. Record the covering PR rather than
treating a loose keyword, a change in the same file, or historical failure output alone as duplicate
evidence.

Two regression fixtures are required:
- #5145 / PR #4958 `PosixPath` serialization: stale because current main already replaced
  `json.dumps(asdict(arm_params))` with the tested subprocess-boundary serializer.
- #5480 / PR #5486 `_run_batch_sequential` 3-tuple return: #5480 was dispatched after PR #5486
  merged under #5482 (not #5480), so an issue-number-only merged-PR search missed it; the covering
  PR is found only by mapping the named failing test
  `test_run_batch_sequential_worker_failure_logs_warning` and its
  `too many values to unpack (expected 2)` signature to current `origin/main`, and the expected
  outcome is `covered/stale` with PR #5486 linked and no implementation worker admitted.

Delegated queue-scout output is only route evidence until verified by the main agent in the target
repository. Before using a scout recommendation to select, claim, or branch for an issue, run:

```bash
uv run python scripts/dev/gh_issue_rest.py thread <number> --repo ll7/robot_sf_ll7
```

This tries the concise `gh issue view --comments` path and falls back to paginated REST only for the
known `repository.issue.projectCards` GraphQL failure. Confirm the URL belongs to
`ll7/robot_sf_ll7`, the issue is still open, ready labels/state are current, recent comments do not
block or supersede the work, and no linked/open PR already covers it. Treat stale state, wrong
repo-owner URLs, omitted recent comments, and duplicate PR coverage as expected scout failure modes.

## Workflow

1. Run preflight for required local tooling and the Scout publication linter:
   `uv run python scripts/dev/check_skills.py --preflight gh-issue-autopilot`.
   Then refresh credentials and branch baseline (`gh auth status`, `git fetch origin`).
2. Resolve the queue candidate and re-check the complete issue thread with
   `uv run python scripts/dev/gh_issue_rest.py thread <number> --repo ll7/robot_sf_ll7` before claim
   or branch, especially when a delegated scout proposed the issue.
3. Check open PRs for duplicate coverage using the linked issue, head branch/scope, and title.
   Stop or update routing when an existing PR covers the work.
4. If issue statement is ambiguous:
   - post a short decision options note,
   - add `decision-required` (create if missing),
   - set status back to `Tracked`,
   - stop with blocker.
5. Acquire the cross-machine issue claim before any implementation branch or worktree setup:

   ```bash
   uv run python scripts/dev/issue_claim.py acquire <issue-number>
   ```

   The command atomically creates the remote ref `agent-claims/issue-<issue-number>` through
   GitHub's create-ref API, which fails when the ref already exists. If it exits non-zero, another
   PC or agent probably claimed the issue first; run
   `uv run python scripts/dev/issue_claim.py status <issue-number>`, record the collision, and skip
   to the next candidate. Do not reimplement the issue unless the remote claim is confirmed stale
   and deliberately released.
6. After a successful claim, make the claim visible in GitHub issue/project state: move the issue to
   `In progress`, add or preserve `state:running` when using state labels, assign the local actor
   when practical, and add a short issue comment naming the claim ref, machine/thread, intended
   implementation branch, and stale-claim cleanup condition.

   Publication rule: use `scripts/dev/gh_comment.sh`, `gh issue/pr comment --body-file`, or REST
   JSON input for Markdown-heavy comments. Do not put bodies containing backticks, YAML, commands,
   or multiline Markdown into inline shell strings. Use REST endpoints for simple label writes when
   `gh` would route through GraphQL, for example:

   ```bash
   gh api repos/ll7/robot_sf_ll7/issues/<number>/labels -f labels[]=state:running
   gh api -X DELETE repos/ll7/robot_sf_ll7/issues/<number>/labels/state:ready
   ```
7. Create the issue branch in a linked worktree for non-trivial implementation. Prefer the
   `AGENTS.md` "Fresh Worktree Bootstrap" location, naming, machine-context symlink, and branch
   freshness rules; use an in-place branch only for tiny or explicitly requested main-checkout work.
8. Implement inside accepted scope.
9. Run validation gate and rerun on failures only when fixable.
10. Commit with conventional message; if long-running benchmark evidence appears, classify artifacts.
11. Sync with latest `origin/main`, rerun readiness, and check artifact durability.
12. Re-check open PRs for the same linked issue, head scope, or title before opening a new PR.
13. Open a ready PR using `gh-pr-opener`; use draft only when explicitly requested or when the
    PR body names a concrete reason review should be blocked.
14. After the PR exists and the issue is visibly covered by the PR, release the transient claim with
    `uv run python scripts/dev/issue_claim.py release <issue-number>`. Do not release the claim
    earlier unless the run is abandoning the issue and records the handoff.
15. For deferred important work, create follow-up issues and link them before final handoff.

## Branch and State Safety

- One active issue branch by default.
- Keep branch names stable and issue-linked.
- Treat `agent-claims/issue-<number>` as the cross-machine mutex for implementation selection. A
  successful claim means this run may proceed; a failed claim means skip the issue, inspect status,
  and choose another candidate. Project status, labels, assignments, and comments are visibility
  surfaces, not the atomic claim.
- Before tearing down a worktree, follow `AGENTS.md` "Worktree Teardown And Preservation" so tracked,
  untracked, and ignored-but-important local changes are preserved or explicitly dismissed.
- Do not force-push, rewrite branch history, or merge unrelated issues into this branch.
- If status/branch drift is detected mid-flow, pause and re-check before continuing.

## Anti-Loop Rules

- Do not cycle between branch implementation and validation on unchanged failures.
- If readiness/auth/project write fails twice, stop and report the blocker with minimal next action.

## Output

Emit:
- selected issue + why,
- branch and PR target,
- commands run,
- artifact classification decision,
- follow-up issues created,
- final PR URL or blocker reason.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.


## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.
