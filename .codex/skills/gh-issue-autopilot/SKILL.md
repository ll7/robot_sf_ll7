---
name: gh-issue-autopilot
description: "Autonomous issue-to-PR workflow with gh: pick next best issue, branch, implement, test with scripts/dev gates, open draft PR, and create/track follow-up issues when scope must be deferred."
---

# GH Issue Autopilot

## Overview

Use this skill when the user asks to "take the next issue" and execute end-to-end from issue
selection to a draft PR, while keeping project metadata and follow-up tracking consistent.

Typical invocation:

- `we have issues here, take the next most reasonable one, implement as you go until draft PR`

## Defaults

- Repository: `ll7/robot_sf_ll7`
- Project: `ll7` Project `#5`
- Base branch: `main`
- Branch prefix: `codex/`
- Validation gate: `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`

## Selection Policy: Next Best Issue

1. Prefer project-backed items with `Status` in this order:
   - `Ready`
   - `Todo`
   - `Tracked`
2. Within same status, prefer `Priority`:
   - `Very High`, `High`, `Medium`, `Low`, `Very Low`
3. Prefer issues without linked PRs and without blocker labels:
   - `decision-required`, `blocked`, `wontfix`, `duplicate`
4. Tie-breaker:
   - lower issue number first (older work first).

Use `gh project item-list` as the source of truth before falling back to plain `gh issue list`.

## Workflow

1. Sync and authenticate
   - `gh auth status`
   - `git fetch origin`
   - `git switch main && git pull --ff-only`

2. Resolve project/field IDs (needed for status and priority updates)
   - `PROJECT_ID=$(gh project view 5 --owner ll7 --format json --jq .id)`
   - `gh project field-list 5 --owner ll7 --format json`
   - Extract:
     - `Status` field ID and option IDs (`Tracked`, `Todo`, `Ready`, `In progress`, `Done`)
     - `Priority` field ID and option IDs (`Very Low` .. `Very High`)

3. Select next candidate issue
   - Query project items:
     - `gh project item-list 5 --owner ll7 --limit 200 --format json`
   - Filter to open issue candidates and rank by policy above.
   - Validate issue is still open:
     - `gh issue view <number> --json state,title,body,labels,milestone,url`

4. Ambiguity check before coding
   - Verify problem statement, acceptance criteria, and boundaries are explicit.
   - If ambiguous, write short options with:
     - `Pros`
     - `Cons`
     - `Recommended option + rationale`
   - If a decision is required from maintainers:
     - ensure label exists:
       - `gh label create "decision-required" --color B60205 --description "Needs maintainer decision" || true`
     - add label: `gh issue edit <n> --add-label decision-required`
     - set project status back to `Tracked`
     - comment options via `scripts/dev/gh_comment.sh issue <n> <<'EOF' ... EOF`
     - stop implementation and report blocking decision.

5. Move issue to active execution
   - Set project `Status` to `In progress`.
   - Optionally set or normalize `Priority` if obviously wrong.

6. Create and checkout issue branch
   - Preferred:
     - `gh issue develop <n> --base main --checkout --name "codex/<n>-<slug>"`
   - Fallback:
     - `git switch -c codex/<n>-<slug>`

7. Implement
   - Keep scope tight to issue acceptance criteria.
   - Add/adjust tests and docs with the code.
   - If scope expands, split deferred work into follow-up issues (step 10).

8. Validate with repository gates
   - `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`
   - If failures occur, fix and rerun until green (or document justified exception).

9. Commit, push, open draft PR
   - Commit in logical batches with conventional messages.
   - `git push -u origin "$(git branch --show-current)"`
   - Build PR body from `.github/PULL_REQUEST_TEMPLATE/pr_default.md`.
   - Open draft PR linking issue:
     - `gh pr create --draft --base main --head <branch> --title "<type>: <summary> (#<n>)" --body-file <prepared_body.md>`

10. Create follow-up issues when needed
   - For deferred but important work, create dedicated issues:
     - `gh issue create --title "<follow-up>" --body-file <body.md> --label "enhancement,technical-debt" --milestone "<milestone>"`
   - Add to project:
     - `gh project item-add 5 --owner ll7 --url <issue_url>`
   - Set `Priority` + `Status` fields on follow-up issue item.
   - Reference follow-up issue IDs in PR description and parent issue comment.

11. Close loop metadata
   - Parent issue:
     - keep open while PR is draft; use closing keyword in PR body (`Closes #<n>`) when ready.
   - Project:
     - issue `Status`: keep `In progress` until merged (or `Done` only if team process wants pre-merge done).
     - PR item `Status`: `In progress` / `Ready for review` if that convention is in use.

## Output Requirements

- Always report:
  - selected issue number + why it was selected
  - ambiguity decisions taken (or why none were needed)
  - commands run for validation
  - follow-up issues created (if any) with labels/milestone/priority
  - draft PR URL
