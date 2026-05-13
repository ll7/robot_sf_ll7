---
name: goal-issue-implementation
description: "Autonomous goal loop that sequentially implements eligible open issues, validates, pushes, and opens PRs."
---

# Goal Issue Implementation

## Overview

Use this skill when the user wants open issues implemented as a goal-driven queue. The default
execution model is sequential: select one eligible issue, create one branch, implement, validate,
push, open one PR, then move to the next eligible issue.

This is an orchestration layer over `gh-issue-sequencer`, `gh-issue-autopilot`,
`implementation-verification`, `pr-ready-check`, `gh-pr-opener`, and `context-note-maintainer`.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `docs/context/goal_driven_agent_loops_2026-05-13.md`
- `docs/context/issue_713_batch_first_issue_workflow.md`
- `.agents/skills/gh-issue-sequencer/SKILL.md`
- `.agents/skills/gh-issue-autopilot/SKILL.md`
- `.agents/skills/implementation-verification/SKILL.md`
- `.agents/skills/pr-ready-check/SKILL.md`

## Preflight

At the start of each goal, state:

- queue source: Project #5, label/milestone filter, explicit issue list, or all open issues,
- write mode: branch creation, issue/project updates, commits, pushes, PR creation, and follow-up
  issues are allowed by default,
- stop condition: eligible queue exhausted, optional time budget reached, unclear issue contract,
  validation blocker, GitHub/auth failure, or user stop,
- exclusions, such as benchmark runs, GPU-only work, blocked issues, or paper-facing claims.

Do not ask for confirmation after this preflight unless the user requested a gated run.

## Eligibility Policy

An issue is eligible when:

- it is open,
- it is not labeled `blocked`, `decision-required`, `duplicate`, `wontfix`, or `invalid`,
- it has no linked open PR that already covers the scope,
- its problem, scope, acceptance criteria, and validation path are clear enough to implement,
- its required execution environment is available or has a documented fallback that does not weaken
  the issue contract.

If eligibility fails, route the issue through `issue-audit`, `gh-issue-clarifier`, or
`gh-issue-template-auditor` instead of guessing.

## Workflow

1. Sequence the queue
   - Use `.agents/skills/gh-issue-sequencer/SKILL.md`.
   - Prefer Project #5 `Ready`, then `Todo`, then `Tracked`, ordered by priority and uncertainty.
   - Keep benchmark/paper-facing blockers ahead of speculative experiments when priorities tie.

2. Execute exactly one issue at a time
   - Use `.agents/skills/gh-issue-autopilot/SKILL.md`.
   - Sync with latest `origin/main` at the start of active work and before PR creation.
   - Create an issue branch with `gh issue develop` when possible.
   - Keep scope tied to the issue contract.

3. Implement and prove
   - Add or update tests/docs with the implementation when needed.
   - Run targeted proof first, then the repository readiness gate:
     `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`
   - For benchmark or planner changes, require real executable evidence and treat fallback or
     degraded execution as a caveat, not success.

4. Handoff through PR
   - Inspect generated `output/` artifacts and classify them as disposable, ignored cache,
     tracked manifest/pointer, or durable-upload required.
   - Commit in logical conventional commits.
   - Push and open a PR with `.github/PULL_REQUEST_TEMPLATE/pr_default.md`.
   - Create follow-up issues for deferred scope before closing the loop.

5. Continue the queue
   - After a PR is open, return to step 1 for the next eligible issue.
   - If the stop condition fires, write a concise handoff with current branch, issue, validation
     state, blockers, and next command.

## Subagent Policy

Use `gpt-5.3-codex-spark` subagents only for bounded side tasks such as file/API lookup, small test
failure summaries, or narrow mechanical edits with clear file ownership. The main agent owns issue
selection, final integration, validation, push, PR creation, and final judgment.

## Guardrails

- Do not process multiple implementation branches in parallel by default.
- Do not close issues before the PR/merge process resolves them unless the issue is explicitly
  obsolete or duplicated and the GitHub update records why.
- Do not let readiness checks from before the latest-main sync count for PR creation.
- Do not depend on worktree-local `output/` contents without a durable source.

## Output Requirements

Report:

- selected issue and why it was eligible,
- branch and PR URL,
- validation commands and results,
- artifact persistence decision,
- follow-up issues created,
- stop condition if the loop ended before queue exhaustion.
