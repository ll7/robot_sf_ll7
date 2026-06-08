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
- Coordination: implementation selection must use the `goal-issue-implementation` issue claim
  protocol (`uv run python scripts/dev/issue_claim.py acquire <issue-number>`) before branching so
  concurrent runs on different PCs do not implement the same issue.

Do not ask for extra confirmation after this preflight.

## Cycle State Machine

Each cycle iteration follows a fixed phase order:

1. `implement` — delegate to `goal-issue-implementation` for one issue.
2. `review` — delegate to `goal-pr-review` for all merge-eligible PRs.
3. `merge` — delegate to `gh-pr-merger` for all `merge-ready` PRs.
4. `discover` — delegate to `goal-issue-discovery` for bounded discovery.

Before each phase, run a delegation checkpoint:

- Choose the routed helper role for the phase when `ai-delegation-routing` is active:
  queue scout, PR blocker reviewer, bounded editor, validation verifier, CI wait monitor, or
  discovery scout.
- Start at least one eligible routed worker or Spark sidecar for any phase likely to exceed about
  10 minutes, unless the next action is a local-only publication step or all routes are unavailable.
- If no helper is used, record `delegation_skipped: <reason>` with one of: `tiny`,
  `critical-path-blocker`, `route-unavailable`, `sensitive-context`, `pure-synthesis`, or
  `local-publication-step`.

Publication and final judgment remain local: delegates must not push, open, or merge PRs; change
labels or project state; resolve review threads; or make final benchmark, paper, or safety claims
unless the user explicitly grants that permission. Their output is `route_evidence` that must be
reviewed and validated before phase completion, not benchmark or claim proof by itself.

### Async CI Wait Policy

Do not idle the main autopilot thread on routine GitHub CI waits when other safe work remains.
When a PR reaches `awaiting_ci` and the local proof bar is otherwise ready:

1. Record the PR number, expected head SHA, current CI state, and static wait budget in the active
   ledger. The default budget is `ceil(920s * 1.3) = 1196s`, unless a newer committed skill/doc
   baseline has replaced it.
2. Start one read-only `ci_wait_monitor` route or Codex app/Spark sidecar for that PR:

   ```bash
   uv run python scripts/dev/watch_pr_ci_status.py <pr-number> \
     --expected-head-sha <head-sha> --json
   ```

3. Continue with non-conflicting work on the main thread: review other PRs, merge already-green
   `merge-ready` PRs, or run bounded discovery. Do not mutate the waiting PR branch or resolve its
   final readiness while its monitor is active.
4. When the monitor returns, the main agent must review the result against the current PR head SHA
   before applying `merge-ready`, merging, or reporting completion.

The wait helper uses the stable default budget on normal runs. It samples recent successful CI
runtime only when CI is still pending after the budget expires, and then reports drift evidence plus
a recommended baseline. Do not change the committed default wait baseline after one ordinary slow
run; update it only after repeated over-budget evidence or an obvious CI workflow change.

Treat monitor exit states as follows:

- `success`: CI is green for the expected head SHA; reassess readiness locally.
- `failure`: leave the PR open, record the failing checks, and continue the cycle.
- `timeout`: keep the PR in `awaiting_ci`, record the drift sample, and continue other work.
- `error`: record stale head, auth, API, or parsing failure; do not trust the waiter for readiness.

### Active Delegation Ledger

For long-running or delegated goal runs, maintain one compact active-state ledger in the common Git
directory so it survives linked worktrees, context compaction, and branch cleanup without entering
the PR diff:

```bash
LEDGER_DIR="$(cd "$(git rev-parse --git-common-dir)" && pwd)/codex-agent-runs/active"
mkdir -p "$LEDGER_DIR"
```

Use a short Markdown or YAML file such as
`$LEDGER_DIR/issue-<number>-delegation-ledger.md` or `$LEDGER_DIR/pr-<number>-delegation-ledger.md`.
This ledger is a handoff checklist, not a replacement for GitHub issues, PR bodies, issue-claim
refs, worker artifacts, validation logs, or final summaries.

Track only the fields needed to resume safely in under one minute:

- route: provider/tool, model or agent role, run ID or artifact path, and route status;
- delegation budget: for long delegated runs, record Gemini attempts, model variant,
  capacity/quota outcome, accepted evidence tier, and any user-defined stop threshold such as
  weekly usage remaining;
- task state: issue/PR number, phase, branch, worktree, claim ref/status, and next action;
- ownership: files or modules the delegate may read or edit;
- validation: commands planned/run, pass/fail state, and any blocker signature;
- PR/CI: PR URL or number, head SHA, review state, CI state, and merge-ready state;
- CI wait: baseline seconds, multiplier, budget seconds, poll interval, deadline, monitor
  route/run ID, expected head SHA, final status, and drift sample when collected;
- cleanup: app-agent or worker close status, claim release status, worktree/artifact decision.

Distinguish route success from task success. A delegate command exiting zero or producing a report
only proves `route_status: completed`; the parent phase is not complete until the main agent has
reviewed the output, integrated any edits, run the required validation, updated GitHub state, and
recorded cleanup.

### Usage Pause Guard

When a user-defined Codex usage threshold is active, treat a `codex-usage-status`
`threshold_decision.status: stop` result as a hard paused state for the autopilot
loop, not as another recoverable phase outcome.

Persist the stop decision in the common Git directory before returning so repeated
automatic continue prompts can short-circuit without rereading repo state, rerunning
GitHub operations, or restarting delegation:

```bash
PAUSE_DIR="$(cd "$(git rev-parse --git-common-dir)" && pwd)/codex-agent-runs/active"
mkdir -p "$PAUSE_DIR"
# Write compact JSON/YAML/Markdown such as:
# $PAUSE_DIR/usage-pause.md
```

The pause record should include the observed timestamp, threshold window,
remaining percentage, threshold percentage, and whether the user may explicitly
override the guardrail.

While the usage pause is active:

- do not run repo, worktree, GitHub, benchmark, validation, or delegation commands;
- do not call usage-check tooling again on automatic "continue" prompts unless a
  recorded cooldown has elapsed or the user explicitly asks for current usage;
- do not load broad skill or repository context just to restate the pause;
- respond to repeated automatic continue prompts with one compact sentence such as
  `Paused: weekly remaining 13% < 28%. No actions.`;
- keep the active goal incomplete unless a real completion audit already proved
  all requirements before the pause fired.

Resume only when the user explicitly overrides the stop guardrail, or when a fresh
usage check requested by the user or allowed by the cooldown reports remaining
budget at or above the threshold. If the first stop check happens while required
cleanup is already in progress, finish only the minimal cleanup or follow-up issue
creation named by the user's guardrail, then enter the persisted paused state.

Update the ledger:

- after a claim is acquired and the implementation worktree/branch is created;
- after each delegated route starts, including `delegation_skipped: <reason>` entries;
- after worker completion, failure, or cancellation;
- after PR creation, before any CI wait, and after CI/review state changes;
- after merge success/failure, claim release, worker close, and worktree/artifact cleanup;
- before any compaction-prone wait, handoff, or final response while active work remains.

Transitions:
- After `implement`, proceed to `review`.
- After `review`, proceed to `merge`.
- After `merge`, proceed to `discover`.
- After `discover`, return to `implement` unless stop condition is met.
- If any phase produces zero work (no eligible issue, no reviewable PR, no
  merge-ready PR, no discovery candidate), record the zero-work outcome and advance.

Do not reorder phases or skip a phase that has eligible work.

### Delegated Lifecycle Cleanup Checkpoint

Before each phase transition, after delegated work is integrated or rejected, run a
lifecycle cleanup checkpoint. Distinguish the two delegate types:

- **Codex app subagents** (GoalIssueImplementationAgent, GoalPRReviewAgent, etc.):
  call `close_agent` with result summary after edits are integrated or the proposal
  is rejected. Do not leave subagent sessions open across phase boundaries.
- **External codex-agent-worker subprocesses**: confirm the subprocess has exited
  (no zombie/lingering process) and verify expected artifacts exist or are explicitly
  discarded. Record `worker completed` with exit code and artifact path or discard
  rationale.

Record cleanup status in the ledger, handoff notes, or self-review companion using
one of:

- `worker completed` — external subprocess exited cleanly, artifacts confirmed.
- `worker_sparse_artifacts` — subprocess exited or was stopped but compact result/status/diffstat
  artifacts are missing or empty; treat as route failure or T0 evidence until local validation
  proves the finding independently.
- `app agent closed` — Codex subagent session closed after integration/rejection.
- `no active process remains` — neither subagent nor worker process remains open.
- `cleanup_failed` — close/confirm failed; record the error and escalate.

A phase is not complete until the cleanup checkpoint passes for every delegate used
in that phase.

## Delegation Failure Recovery

Each delegate skill may fail. Handle failures per phase:

- `implement` failure:
  - If issue-claim acquisition fails: classify the issue as already claimed/running, record
    `uv run python scripts/dev/issue_claim.py status <issue-number>` output, skip the issue, and
    continue to the next candidate. Do not branch or make local edits for that issue.
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
