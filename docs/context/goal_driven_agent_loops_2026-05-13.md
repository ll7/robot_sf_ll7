# Goal-Driven Agent Loop Skills

**Date**: 2026-05-13
**Status**: Adopted as repo-local skill-first workflow guidance

## Purpose

This note records the shared contract for four goal-driven agent loop skills:

- `.agents/skills/goal-issue-discovery/SKILL.md`
- `.agents/skills/issue-audit/SKILL.md`
- `.agents/skills/goal-issue-implementation/SKILL.md`
- `.agents/skills/goal-pr-review/SKILL.md`

The loops are intentionally skill-first. No slash-command wrappers, prompt wrappers, or formal
`docs/superpowers/specs/` design document are part of the initial implementation.

## Shared Contract

Each autonomous goal loop starts with a short preflight that states the active loop, scope, write
mode, stop condition, and exclusions. After that preflight, autonomous GitHub writes are allowed by
default. This includes issue creation and edits, issue consolidation or splitting, Project #5
updates, branch/PR work, follow-up issue creation, and `merge-ready` labeling when the written
criteria pass.

The default stop condition is queue exhaustion. A user-provided time budget is also valid; when a
time budget stops the loop, the agent must leave a concise handoff that names the last completed
item, current blocker, changed GitHub objects, and next recommended action.

Goal-loop skills are orchestration layers, not replacements for narrower workflow skills. They
should delegate to existing repo-local skills for concrete work:

- issue creation: `gh-issue-creator`
- issue clarification and template repair: `gh-issue-clarifier`, `gh-issue-template-auditor`
- queue routing: `gh-issue-sequencer`, `gh-issue-priority-assessor`
- issue implementation: `gh-issue-autopilot`, `gh-pr-opener`
- PR repair during review: `gh-pr-comment-fixer`, `gh-issue-creator`
- verification: `implementation-verification`, `pr-ready-check`, `review-benchmark-change`
- durable notes: `context-note-maintainer`

Spark subagents may be used only for bounded side tasks such as locating files, summarizing a small
area, inspecting command output, or performing narrow mechanical edits with clear file ownership.
The main agent remains responsible for planning, integration, validation, GitHub writes, and final
judgment.

## Loop Boundaries

`goal-issue-discovery` may create broad proposal issues, not only observed bug reports. Every new
issue must carry an evidence grade: `observed`, `inferred`, or `proposal`. The grade prevents
speculative ideas from being presented as measured defects or benchmark regressions.

`issue-audit` is not a default `/goal` loop. It is a user-in-the-loop refinement workflow that
orders open issues by readiness blockers and asks exactly one user-facing question at a time. It
updates issue bodies, labels, comments, Project #5 metadata, and follow-up issues as decisions are
made.

`goal-issue-implementation` processes issues sequentially. It selects one eligible issue, creates
one branch, implements, validates, checks generated artifacts, commits, pushes, and opens one PR
before continuing. It stops instead of guessing when an issue is blocked, ambiguous, already covered
by an open PR, or missing proof requirements.

`goal-pr-review` reviews open PRs against linked issue contracts, not just CI state. When the full
proof bar fails, the default path is fix-first if the PR branch is writable and the smallest repair
stays inside the issue or PR contract. The loop should classify each gap as auto-fixable now,
deferred follow-up, or handoff-only blocker; repair safe gaps on-branch; rerun targeted proof plus
the readiness gate; then reassess `merge-ready`. It applies a dedicated `merge-ready` label only
after the full proof bar passes: issue contract resolved, diff matches scope, checks or readiness
proof are adequate, review threads are handled, generated artifacts are classified, and deferred
work is captured as follow-up issues.

## Deferred Scope

The initial implementation deliberately avoids:

- slash-command or prompt-wrapper aliases,
- a central router skill that guesses which loop to run,
- persistent daemon-style execution,
- parallel issue implementation as the default,
- a new database, retrieval layer, or queue service.

Those can be reconsidered after the skill-first loops prove useful and the failure modes are clear.

## Validation Surface

Changes to these skills should be validated with:

```bash
uv run python scripts/dev/check_skills.py
uv run python scripts/tools/sync_ai_config.py --check
```

For docs/skill-only changes, also check discoverability by grepping for the skill names in
`.agents/skills/README.md`, `docs/context/README.md`, and any touched AI entry points. Before PR
handoff, run the repository readiness gate unless there is a documented reason to use a narrower
proof:

```bash
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

## Related

- `docs/context/issue_713_batch_first_issue_workflow.md`
- `docs/context/issue_728_coding_agents_compatibility.md`
- `docs/ai/awesome_copilot_adaptation.md`
- `.agents/skills/README.md`
