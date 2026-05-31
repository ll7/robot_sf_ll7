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

## Workflow

1. Refresh credentials and branch baseline (`gh auth status`, `git fetch origin`).
2. Resolve queue candidate and re-check issue state.
3. If issue statement is ambiguous:
   - post a short decision options note,
   - add `decision-required` (create if missing),
   - set status back to `Tracked`,
   - stop with blocker.
4. Move the issue to `In progress` (optionally normalize priority).
5. Create issue branch (`gh issue develop` preferred; fallback to git branch).
6. Implement inside accepted scope.
7. Run validation gate and rerun on failures only when fixable.
8. Commit with conventional message; if long-running benchmark evidence appears, classify artifacts.
9. Sync with latest `origin/main`, rerun readiness, and check artifact durability.
10. Open a ready PR using `gh-pr-opener`; use draft only when explicitly requested or when the
    PR body names a concrete reason review should be blocked.
11. For deferred important work, create follow-up issues and link them before final handoff.

## Branch and State Safety

- One active issue branch by default.
- Keep branch names stable and issue-linked.
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
