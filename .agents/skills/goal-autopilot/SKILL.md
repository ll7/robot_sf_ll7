---
name: goal-autopilot
description: Continuous goal autopilot; orchestrates implement, review, merge, and discover cycles
  with preflight validation and delegation failure recovery.
category: general
kind: orchestrator
phase: implementation
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to:
- goal-issue-implementation
- gh-pr-merger
- goal-pr-review
- goal-issue-discovery
output_schema: skill_run_summary.v1
aliases:
- continuous-autopilot
- implement-review-merge-discover
---

# Goal Autopilot

Use this skill for a continuous issue-to-merge-to-discovery loop.

It orchestrates:
- `goal-issue-implementation` — select, implement, validate, and open PRs.
- `goal-pr-review` — review, fix, and apply `merge-ready`.
- `gh-pr-merger` — merge approved PRs.
- `goal-issue-discovery` — discover new improvement opportunities.

It does not define child-skill mechanics; it standardizes cycle policy, preflight
validation, and delegation failure recovery across the loop.

## Trigger Boundary

Use this skill when the user asks for a continuous autopilot across the full
implement → review → merge → discover cycle.

Do not use it for:
- single-issue implementation without review/merge,
- passive PR review without merge authority,
- standalone issue discovery without follow-through,
- manual one-off PR merges.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `docs/context/goal_driven_agent_loops_2026-05-13.md`
- `docs/context/issue_713_batch_first_issue_workflow.md`
- `.agents/skills/goal-issue-implementation/SKILL.md`
- `.agents/skills/goal-pr-review/SKILL.md`
- `.agents/skills/gh-pr-merger/SKILL.md`
- `.agents/skills/goal-issue-discovery/SKILL.md`

## Preflight

Run before the first cycle iteration:

```bash
uv run python scripts/dev/check_skills.py --preflight goal-autopilot
```

The `--preflight` check reports:
- declared runtime requirements from `.agents/skills/skills.yaml`,
- local command availability for `gh`, `git`, and `uv`,
- Project #5 readiness at the GitHub CLI/tooling level.

It does not replace phase-specific label, branch-protection, review-thread, or CI
checks. Run those inside the delegated phase that owns the operation.

If preflight fails, stop and report the blocker with the failing check name and exit code.
Do not retry preflight without fixing the identified gap.

Record at start:
- Cycle scope: eligible issues, open PRs, and discovery lanes.
- Write permissions: branch/commit/PR/project/merge writes allowed by default.
- Stop condition: all eligible issues processed, no merge-ready PRs remain, discovery
  saturated, or user stop.
- Exclusions: blocked/decision-required issues, draft PRs, benchmark-heavy PRs that
  need manual review.

Do not ask for extra confirmation after this preflight.

## Cycle State Machine

Each cycle iteration follows a fixed phase order:

1. `implement` — delegate to `goal-issue-implementation` for one issue.
2. `review` — delegate to `goal-pr-review` for all merge-eligible PRs.
3. `merge` — delegate to `gh-pr-merger` for all `merge-ready` PRs.
4. `discover` — delegate to `goal-issue-discovery` for bounded discovery.

Before each phase, run a delegation checkpoint:

- Choose the routed helper role for the phase when `ai-delegation-routing` is active:
  queue scout, PR blocker reviewer, bounded editor, validation verifier, or discovery scout.
- Start at least one eligible routed worker or Spark sidecar for any phase likely to exceed about
  10 minutes, unless the next action is a local-only publication step or all routes are unavailable.
- If no helper is used, record `delegation_skipped: <reason>` with one of: `tiny`,
  `critical-path-blocker`, `route-unavailable`, `sensitive-context`, `pure-synthesis`, or
  `local-publication-step`.

Publication and final judgment remain local: delegates must not push, open, or merge PRs; change
labels or project state; resolve review threads; or make final benchmark, paper, or safety claims
unless the user explicitly grants that permission. Their output is route evidence that must be
reviewed and validated before phase completion.

Transitions:
- After `implement`, proceed to `review`.
- After `review`, proceed to `merge`.
- After `merge`, proceed to `discover`.
- After `discover`, return to `implement` unless stop condition is met.
- If any phase produces zero work (no eligible issue, no reviewable PR, no
  merge-ready PR, no discovery candidate), record the zero-work outcome and advance.

Do not reorder phases or skip a phase that has eligible work.

## Delegation Failure Recovery

Each delegate skill may fail. Handle failures per phase:

- `implement` failure:
  - If the issue is ambiguous: route to `issue-contract-maintainer`, mark skipped.
  - If validation fails twice: mark issue `blocked`, record the failing command
    and last error, and continue to the next eligible issue.
  - If branch/push collision: record the collision details, skip the issue,
    and continue. Do not force-push.

- `review` failure:
  - If the PR is blocked externally: leave it in `awaiting_reviewer`, continue.
  - If fix + push fails: record the failure signature, move PR to `blocked_external`.
  - If `merge-ready` cannot be applied because proof bar is not met: leave PR
    at current state, report the gap, continue.

- `merge` failure:
  - If merge conflict: report conflict, leave PR open, continue.
  - If CI status check fails: leave PR in `merge-ready` (CI is async), report
    the failing check, continue.
  - If branch protection rejects: record rejection reason, continue.
  - If `gh` CLI merge fails due to auth/permission: stop the autopilot and report
    the auth blocker.

- `discover` failure:
  - If API writes fail: emit handoff with partial results, continue.
  - If duplicate creation: skip candidate, log rationale, continue.
  - If Project #5 write failure: batch writes once, skip lane if quota exhausted.

Do not let one phase failure block the entire cycle unless it is an auth or
environment blocker that will affect all phases.

### Agent Run Self-Review

When a delegated phase produces a reusable workflow lesson (repeated failure pattern,
tooling gap, routing improvement), capture it in an `agent_run_self_review.v1`
companion note before the next phase begins. Do not promote lessons into durable
skill text unless the evidence is repeated or explains a costly failure.

## Cycle Policy

- Process exactly one implementation issue per cycle iteration by default.
- After implementation, review all open non-draft PRs that are not blocked.
- Merge all PRs carrying the `merge-ready` label.
- Run one bounded discovery pass after merge.
- End the cycle when all three produce zero work.

## Stop Conditions

Stop the autopilot when any:
- all eligible issues have been implemented and merged,
- no new discovery candidates remain,
- auth/credentials/env blocker that affects all phases,
- user requests stop.

Record a final summary: phases executed, issues implemented, PRs merged,
discovery issues created, and any blockers encountered.

## Confidence

- `High`: all four phases completed successfully with proof.
- `Medium`: some phases produced zero work or minor failures were recovered.
- `Low`: auth/environment blocker halted the cycle.

## Required Output

For each cycle iteration, report:
- phase executed and delegate skill used,
- issue/PR number and outcome,
- validation commands and pass/fail results,
- delegation failure and recovery action,
- follow-up issues or deferred work,
- final confidence and stop reason.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.


## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.


## Output

Return the schema named by the `output_schema` frontmatter field, or a compact equivalent when the caller does not require YAML.
