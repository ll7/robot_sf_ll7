---
name: gh-pr-opener
description: "Open Robot SF PRs with MCP-first GitHub interaction, explicit issue-scope verification, and a conservative PR-readiness freshness gate."
---

# GH PR Opener

## Overview

Use this skill when a branch is ready to be handed off as a GitHub PR and the repository wants a
standard, conservative opening flow.

Prefer GitHub MCP / GitHub app tools for interactive issue and PR inspection when available.
Keep the concrete `gh` commands below as the deterministic fallback for PR creation, readiness
verification, and metadata operations that still need CLI scripting.

This skill is fail-closed by design:

- It does **not** open a PR until the linked issue scope appears implemented.
- It does **not** trust old validation blindly.
- It uses `.github/PULL_REQUEST_TEMPLATE/pr_default.md` as the PR body base.

## Freshness Rule

A prior readiness run counts as fresh only when all of the following are true:

- the local stamp file for the current branch exists,
- `status == "passed"`,
- the stamped branch matches the current branch,
- the stamped `head_sha` matches `git rev-parse HEAD`,
- the stamped `base_ref` matches the intended PR base (default `origin/main`),
- and the stamp is at most 24 hours old.

Canonical helper:

- `uv run python scripts/dev/pr_ready_freshness.py status --base-ref origin/main`

If freshness cannot be proven, rerun:

- `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`

After a successful rerun, record freshness evidence:

- `uv run python scripts/dev/pr_ready_freshness.py write --base-ref origin/main`

## Workflow

1. Confirm branch + issue context
   - `git branch --show-current`
   - `gh issue view <n> --json number,title,body,url,state`
   - Stop if the issue is closed, superseded, or the branch is unrelated.

2. Verify the issue scope is actually implemented
   - Read the issue goal, scope, acceptance criteria, and definition of done.
   - Review `git diff origin/main...HEAD --stat` and targeted file diffs.
   - Check that each in-scope requirement is either implemented now or explicitly deferred into a
     linked follow-up issue.
   - If the issue is only partially done, do not open the PR yet.

3. Sync with latest main before opening the PR
   - `git fetch origin main`
   - Merge or rebase the latest `origin/main` into the feature branch before PR creation.
   - Treat any readiness result from before the latest-main sync as stale.

4. Verify readiness freshness after the sync
   - Run:
     - `uv run python scripts/dev/pr_ready_freshness.py status --base-ref origin/main`
   - If the helper exits non-zero or returns non-fresh JSON, rerun:
     - `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`
     - `uv run python scripts/dev/pr_ready_freshness.py write --base-ref origin/main`

5. Build the PR body from the repository template
   - Start from `.github/PULL_REQUEST_TEMPLATE/pr_default.md`.
   - Fill sections with repository-specific evidence:
     - summary of implementation,
     - linked issue (`Closes #<n>` when appropriate),
     - validation commands actually run,
     - risks/limitations,
     - docs/provenance references,
     - follow-up issues if any.
   - Do not remove sections; keep the template structure intact.

6. Open the draft PR
   - Preferred command:
     - `gh pr create --draft --base main --head <branch> --title "<type>: <summary> (#<n>)" --body-file <prepared_body.md>`
   - Use a conventional commit/PR title prefix such as `feat:`, `fix:`, or `docs:`.

7. Close the loop
   - Keep the parent issue open while the PR is draft unless repository policy says otherwise.
   - Comment on the issue or PR when a follow-up item was intentionally deferred.

## Required Evidence

- Current issue scope reviewed against the actual branch diff.
- Latest `origin/main` integrated into the feature branch before PR creation.
- Fresh readiness proof from either:
  - a passing `pr_ready_freshness.py status`, or
  - a new `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` run plus a new stamp.
- PR body based on `.github/PULL_REQUEST_TEMPLATE/pr_default.md`.

## Notes

- The readiness stamp is local branch evidence under `output/validation/pr_ready/`; this keeps the
  rule explicit and reproducible without changing `scripts/dev/pr_ready_check.sh`.
- Conservative behavior is preferred over skipping validation by mistake.
