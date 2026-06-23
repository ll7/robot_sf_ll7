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

## Default Token-Efficient Mode

When the user asks for goal autopilot, a continuous goal loop, or best-quality
repo progress with low native Codex token use, compose this skill with
`save-codex-tokens` by default unless the user explicitly disables token
conservation.

Use this division of responsibility:

- `goal-autopilot` owns the implement -> review -> merge -> discover loop,
  issue claiming, phase transitions, active ledgers, CI wait handoff, cleanup,
  and final readiness/publication decisions.
- `save-codex-tokens` owns budget checks, route selection, delegation
  economics, compact parent-thread snapshots, compact worker evidence, artifact
  review order, and token-budget stop behavior.

In token-efficient mode:

- Codex is the loop controller, reviewer, and final acceptance authority.
- Delegate workers only when compact review cost is lower than direct Codex
  execution.
- On every resume, compaction recovery, interruption return, or new user
  message, first run a current-request guard: compare the newest request with
  the active ledger action, identify active worktree/PR/issue/delegates, and
  decide whether the previous goal action is still the current objective. If
  the newest request supersedes it, park the old batch with a compact handoff
  and cleanup status before editing or opening a new PR for the new request.
- On continuation or after context compaction, rebuild state from the active
  ledger and compact PR/issue/worktree snapshots first. Reopen full skills,
  docs, raw CI output, or broad GitHub/worktree inventories only when the
  ledger is missing, stale, or lacks a field needed for the next mutation.
  Record the loaded-context cache in the active profile so repeated resumes do
  not reread long skill/docs surfaces that are already fresh for the current
  branch, issue/PR head SHA, and phase.
- Prefer compact parent-thread snapshots before broad repository, GitHub,
  worktree, CI, or validation output.
- For routine orientation, start with `scripts/dev/autopilot_state_snapshot.py`
  instead of repeated broad `gh issue list`, `gh pr view`, `git worktree list`,
  or claim-state calls. Read its `controller_checkpoint.token_efficiency`
  recommendations before deciding whether a broader command is still needed.
- Require artifact-first compact worker artifacts before reading raw logs.
- For every delegated implementation/review/queue task, the worker **must** write compact artifacts in
  `<run_artifact_dir>` as: `result.json`, `RESULT.md`, `diffstat.txt`, and `validation.json`.
  Parent review order is mandatory:
  1. Read `result.json`.
  2. Read `RESULT.md` for narrative and risk summary.
  3. Read `diffstat.txt`.
  4. Run only targeted local read/validation (e.g., `git diff <file>`, `git show <commit>`, quick
     rechecks) against the committed candidate state.
- Only then, and only if needed, read raw logs, CI text, or verbose worker transcripts when artifacts are
  missing, inconsistent, failed, or suspicious.
- Worker prompts must cap parent-thread raw output at about 200 lines. Use `rg -l`, `rg --files`,
  and bounded `sed -n` ranges for discovery. Do not use broad `rg -n .` or full file reads unless
  a context pack, snapshot, or targeted artifact is missing and the broad read is explicitly
  justified. Redirect larger command output to private artifacts and return only a compact summary
  with inspected files, command exit codes, and short evidence excerpts.
- If a command is expected to exceed that cap, run it through
  `scripts/dev/run_compact_validation.py`, a purpose-built snapshot helper, or a
  private common-Git-dir artifact and paste only the path plus the short
  evidence needed for the next controller decision.
- A successful worker command exit code, positive wrapper message, or candidate commit is route evidence
  only. The parent phase is not complete until this evidence is reviewed, diff status is verified locally,
  and required commands are rerun from the parent with parent-owned proof.
- Offload routine CI waits to read-only monitors when safe work remains, but
  give monitors a local wall-clock cap and stop them before patching or pushing
  the watched branch.
- Keep final GitHub mutation, publication, merge-readiness, benchmark, paper,
  and safety decisions local.

The default user prompt can be short:

```text
Run goal autopilot for this repo.

Optimize for the best-quality, highest-value eligible work with the least
practical native Codex token usage. Use goal-autopilot as the loop controller
and compose it with save-codex-tokens for routing, budget checks, compact
evidence, and delegation economics.

Continue implement -> review -> merge -> discover until no eligible work
remains, a hard blocker appears, or the Codex budget guard fires.
```

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
- Before broad issue or PR queue review, prefer compact parent-thread snapshots:
  `uv run python scripts/dev/snapshot_issue_batch.py --claimable --limit <n> --json` for a
  no-arg next-issue queue, `uv run python scripts/dev/snapshot_issue_batch.py <first> <last> --json`
  for explicit issue batches, `uv run python scripts/dev/snapshot_pr_queue.py --active --limit <n>
  --json` for the active PR queue, and `uv run python scripts/dev/snapshot_pr_queue.py --prs <pr>
  [<pr> ...] --json` for explicit PR headline state. Apply
  `uv run python scripts/dev/pr_loop_policy.py --snapshot <queue.json> --json` for
  machine-checkable state classification and next-action decisions under a loop budget. Use
  `--capsule-dir <private-artifact-dir>` when an implementation worker should receive a bounded
  issue context capsule instead of rediscovering files with broad search.
- For optional discovery scouts, default to a short hard timeout (120-180s) and require periodic
  evidence in the ledger. If a bounded scout emits no heartbeat within one timeout slice, treat it as
  incomplete and retry with an explicit local timeout + heartbeat plan.
- Start at least one eligible routed worker or Spark sidecar for any phase likely to exceed about
  10 minutes, unless the next action is a local-only publication step or all routes are unavailable.
- Before spawning a Spark sidecar, check the active ledger or most recent route
  snapshot for a Spark quota reset/usage-limit marker. If Spark is unavailable,
  record the reset time and route directly to the next eligible cheap worker
  instead of discovering the quota limit by failed spawn. If a spawn still
  returns a usage-limit error, close the app-agent handle immediately, add the
  reset time to the route cache, and continue locally or with the next eligible
  route rather than retrying Spark.
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
2. Start one read-only `ci_wait_monitor` route or Codex app/Spark sidecar for that PR. In non-TTY
   agent sessions, use bounded polling rather than `gh pr checks --watch`:

   ```bash
   scripts/dev/run_worktree_shared_venv.sh -- python scripts/dev/check_pr_ci_status.py <pr-number> \
     --expected-head-sha <head-sha> \
     --poll-attempts 40 \
     --poll-interval 30 \
     --max-wall-seconds 1200 \
     --json
   ```

   For a non-blocking current-state snapshot, prefer:

   ```bash
   uv run python scripts/dev/watch_pr_ci_status.py <pr-number> \
     --expected-head-sha <head-sha> \
     --json \
     --once
   ```

   For longer delegated monitor waits, add `--emit-progress-json-every <seconds>` so the worker
   returns compact progress evidence instead of silent polling. Do not leave
   long JSON poll streams in the parent thread when no decision changes; store
   verbose progress in the ledger or private artifacts and surface only check
   counts, failures, stale-head state, and terminal status.

3. Continue with non-conflicting work on the main thread: review other PRs, merge already-green
   `merge-ready` PRs, or run bounded discovery. Do not mutate the waiting PR branch or resolve its
   final readiness while its monitor is active.
4. When the monitor returns, the main agent must review the result against the current PR head SHA
   before applying `merge-ready`, merging, or reporting completion.

The polling helper prints queued, in-progress, failed, and passed check summaries. With `--json`,
each poll payload includes compact `monitor` metadata: expected head SHA, SHA-match result, attempt
count, poll interval, wait budget, deadline, and `route_evidence_only: true`. Review that compact
payload before considering raw CI logs or full command output. Use
`gh run view <run-id> --json status,conclusion,jobs` when a job URL needs deeper state, and fetch
logs only after the relevant job has completed. Do not change committed polling budgets after one
ordinary slow run; update them only after repeated over-budget evidence or an obvious CI workflow
change.

Treat monitor exit states as follows:

- `success`: CI is green for the expected head SHA; reassess readiness locally.
- `failure`: leave the PR open, record the failing checks, and continue the cycle.
- `pending timeout` / exit code `2`: keep the PR in `awaiting_ci`, record the pending checks, and
  continue other work.
- `error`: record stale head, auth, API, or parsing failure; do not trust the waiter for readiness.

### Snapshot-First Parent Orientation

Before broad queue, worktree, claim, PR, or CI reads in the parent thread, prefer the compact
snapshot helpers. Use the full-state `autopilot_state_snapshot.py` for initial orientation, then
use targeted compact snapshots for specific needs:

```bash
# Full orientation snapshot (worktrees, claims, issues, PRs)
uv run python scripts/dev/autopilot_state_snapshot.py \
  --include-worktrees \
  --claim-issue <issue-number> \
  --issue-search "is:issue is:open <queue-filter>" \
  --pr <pr-number>

# Compact worktree bootstrap-state snapshot (fresh worktrees, local.machine.md, .venv)
uv run python scripts/dev/compact_worktree_snapshot.py --filter <issue-or-branch-slug> --json

# Compact CI snapshot for PR queue (check rollup, optional drift sample)
uv run python scripts/dev/compact_ci_snapshot.py <pr> [<pr> ...] \
  --expected-head-sha <sha> --json [--include-drift]

# Compact no-arg next-issue queue and explicit issue batch snapshots
uv run python scripts/dev/snapshot_issue_batch.py --claimable --limit <n> --json
uv run python scripts/dev/snapshot_issue_batch.py <first> <last> \
  --json --capsule-dir <artifact-dir>

# Compact active PR queue and explicit PR headline snapshots
uv run python scripts/dev/snapshot_pr_queue.py --active --limit <n> --json
uv run python scripts/dev/snapshot_pr_queue.py --prs <pr> [<pr> ...] --json

# Machine-checkable PR loop policy (state + next action under budget)
uv run python scripts/dev/pr_loop_policy.py --snapshot <queue-snapshot.json> --json
```

Use `compact_worktree_snapshot.py` before expensive commands to detect fresh worktrees that need
bootstrap. Prefer `--filter <issue-or-branch-slug>` so large worktree fleets do not re-enter the
parent thread. The `is_fresh` and `bootstrap_required` fields indicate whether `local.machine.md`
symlink and `uv sync` steps are needed.

Use `compact_ci_snapshot.py` for token-efficient PR CI monitoring. The optional `--include-drift`
flag adds recent successful run timings to recommend an updated wait budget after timeout. Pass
`--expected-head-sha` whenever the parent has a known PR head so stale snapshots fail closed before
CI or merge-readiness decisions.

Treat all snapshot output as **route evidence only**. Run fresh local checks before issue claim,
push, PR publication, label/project mutation, merge-ready application, merge, or benchmark/paper-facing
publication decisions.

The helper emits `autopilot_state_snapshot.v1` JSON with source commands, branch/head SHA,
`origin/main` SHA, worktree rows, issue queue rows, claim refs, explicit PR headline state, and
freshness metadata. Treat it as route evidence only: use it to decide the next safe read or worker
prompt, then run fresh local/GitHub checks before claiming an issue, pushing, labeling, merging, or
publishing a benchmark-facing conclusion. Read raw command output only when the snapshot reports
`ok: false`, stale claim refs, missing state, or a field that is insufficient for the next decision.
The helper caps worktree rows by default and reports `worktree_count` plus `worktrees_truncated`;
raise `--worktree-limit` only when the compact rows are insufficient.

### Active Delegation Ledger

For long-running or delegated goal runs, maintain one compact active-state ledger in the common Git
directory so it survives linked worktrees, context compaction, and branch cleanup without entering
the PR diff:

```bash
LEDGER_DIR="$(git rev-parse --path-format=absolute --git-common-dir)/codex-agent-runs/active"
mkdir -p "$LEDGER_DIR"
```

Use a short Markdown or YAML file such as
`$LEDGER_DIR/issue-<number>-delegation-ledger.md` or `$LEDGER_DIR/pr-<number>-delegation-ledger.md`.
When a phase produces compact snapshots, also record their paths in the ledger and pass the fresh
ledger snapshot paths or context capsules to the next worker prompt instead of asking the worker to
rediscover the same queue, PR, CI, or worktree state. This ledger is a handoff checklist, not a
replacement for GitHub issues, PR bodies, issue-claim refs, worker artifacts, validation logs, or
final summaries.

Track only the fields needed to resume safely in under one minute:

- current request: newest user request, whether it supersedes the prior ledger
  action, and the parked-work handoff path when a pivot occurs;
- route: provider/tool, model or agent role, run ID or artifact path, and route status;
- loaded context: skill/doc summaries already read for the active phase, plus the freshness keys
  that make them reusable or stale;
- snapshots: issue, PR, worktree, claim, and CI snapshot paths with freshness keys such as issue
  number, PR number, branch, origin/main SHA, head SHA, expected PR head SHA, captured-at time, and
  source helper command;
- delegation budget: for long delegated runs, record Gemini attempts, model variant,
  capacity/quota outcome, accepted evidence tier, and any user-defined stop threshold such as
  weekly usage remaining;
- task state: issue/PR number, phase, branch, worktree, claim ref/status, and next action;
- ownership: files or modules the delegate may read or edit;
- validation: commands planned/run, pass/fail state, and any blocker signature;
- PR/CI: PR URL or number, head SHA, review state, CI state, and merge-ready state;
- CI wait: baseline seconds, multiplier, budget seconds, poll interval, deadline, monitor
  route/run ID, expected head SHA, final status, and drift sample when collected;
- workers: current worker run IDs, worker artifact paths, artifact status, and accepted/rejected
  evidence tier;
- cleanup: app-agent or worker close status, claim release status, worktree/artifact decision;
- stale-state triggers: missing or mismatched claim refs, issue/PR state changes, branch/head SHA
  drift, expected PR head SHA drift, stale origin/main, CI rerun/retry, failed validation,
  interrupted worker, missing compact artifacts, or elapsed freshness window.

Consult the ledger before repeating broad state polling, full skill/doc reads, or worker
rediscovery. Repeated broad `gh issue list`, `gh pr view`, `git worktree list --porcelain`, CI log
fetches, full skill reads, or repository-wide search after a fresh ledger snapshot exists is an
anti-pattern unless a stale-state trigger fired or the compact snapshot lacks the field needed for
the next decision. Fresh live checks are still required before issue claim, push, PR publication,
label/project mutation, merge-ready application, merge, claim release, or any benchmark/paper-facing
publication decision.

If the user switches from a goal loop to an instruction, review, cleanup, or
single-PR request, do not continue implementing the old issue just because the
ledger has a next action. Record the old worktree, dirty status, active PRs,
subagent lifecycle state, and next safe resumption command in the ledger, then
scope the new branch and PR to the new request only.

Distinguish route success from task success. A delegate command exiting zero or producing a report
only proves `route_status: completed`; the parent phase is not complete until the main agent has
reviewed the output, integrated any edits, run the required validation, updated GitHub state, and
recorded cleanup.

For delegated queue scouts, keep the scout output in the ledger as route evidence only until the
main agent verifies issue state in `ll7/robot_sf_ll7` with local `gh issue view` or REST evidence.
Do not claim or branch from scout text alone; stale state, wrong repo-owner URLs, missing recent
comments, and duplicate PR coverage are known failure modes.

### Usage Pause Guard

When a user-defined Codex usage threshold is active, run the configured usage-check command. For the
external `codex-usage-status` skill, use the skill's `read_codex_usage.py` helper with
`--stop-below-remaining <percent> --json`. That helper returns a `threshold_decision` object with
`status: continue`, `status: stop`, or `status: unavailable`. Handle every status explicitly:

- `continue`: proceed with the next autopilot phase without creating or refreshing a pause record.
- `stop`: enter a hard paused state for the autopilot loop, not another recoverable phase outcome.
- `unavailable`: treat missing, malformed, or incomplete `threshold_decision` output as uncertain;
  log the raw helper output in the ledger, do not create a hard pause record, and continue only with
  conservative cleanup or handoff work unless the user explicitly permits a normal phase to proceed.

Persist the stop decision in the common Git directory before returning so repeated
automatic continue prompts can short-circuit without rereading repo state, rerunning
GitHub operations, or restarting delegation:

```bash
PAUSE_DIR="$(git rev-parse --path-format=absolute --git-common-dir)/codex-agent-runs/active"
mkdir -p "$PAUSE_DIR"
# Write compact JSON/YAML/Markdown such as:
# $PAUSE_DIR/usage-pause.md
```

The pause record should include the observed timestamp, threshold window, remaining percentage,
threshold percentage, whether the user may explicitly override the guardrail, `lastUsageCheckAt`,
and `usageCheckCooldownSeconds`. The default cooldown is 900 seconds. A caller may set a longer
cooldown in the pause record, but automatic "continue" prompts must compare the current time against
`lastUsageCheckAt + usageCheckCooldownSeconds` before invoking usage-check tooling again.

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

For discovery scouts and other optional helper runs, record periodic heartbeat events in the same
ledger line set using stable fields:

- `timestamp`: UTC timestamp of the heartbeat.
- `phase`: current execution phase.
- `duration`: bounded duration covered by the heartbeat tick.

Record cleanup status in the ledger, handoff notes, or self-review companion using
one of:

- `worker completed` — external subprocess exited cleanly, artifacts confirmed.
- `worker_sparse_artifacts` — subprocess exited or was stopped but compact result/status/diffstat
  artifacts are missing or empty; treat as route failure or T0 evidence until local validation
  proves the finding independently. For `worker_sparse_artifacts`, the parent route is not complete
  until a local re-validation command is recorded and passes (for example: issue/PR state recheck,
  command re-run, deterministic diff check, or scripted validator).
- `app agent closed` — Codex subagent session closed after integration/rejection.
- `no active process remains` — neither subagent nor worker process remains open.
- `cleanup_failed` — close/confirm failed; record the error and escalate.

A phase is not complete until the cleanup checkpoint passes for every delegate used
in that phase.

Run the cleanup checkpoint again before honoring a user pivot or final handoff.
The result should state `closed`, `preserved`, or `not_applicable` for each
subagent/worker, and should explain any dirty worktree left behind.

### Spark Sidecar Routing

Spark (`gpt-5.3-codex-spark`, or the configured Spark sidecar model) is a first-class route for
small, low-risk read-only task classes during the autopilot cycle. Route Spark when the task fits
one of:

- **tiny lookup** — file location, name resolution, short grep.
- **read-only review** — narrow diff inspection, single-file summary.
- **docs cross-check** — link validation, path reference checks.
- **issue/file surface mapping** — issue-to-file coverage, surface enumeration.
- **inspect small command output** — bounded stdout/stderr review.

Spark prompts must require compact output: files inspected, exact evidence, uncertainty, and
recommended next prompt.

When Spark is rate-limited, treat the failed spawn as a routing signal, not as
task evidence. Close the app-agent handle when one was allocated, cache the
reset timestamp in the active ledger, and skip Spark for the rest of the phase.

Do not route Spark to:

- final benchmark interpretation and paper claims,
- merge readiness and publication decisions,
- GitHub mutation (labels, comments, PR creation, merge, close),
- long CI polling unless a bounded monitor helper exists,
- shell-executable fallback unless a real headless wrapper is available.

This is routing guidance only; do not configure Spark as a shell-executable fallback.

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

If the phase used `worker_sparse_artifacts`, require the self-review companion to also log:
- Missing artifact class (`result`, `status`, `diffstat`, or `summary`).
- Scout timeout and heartbeat evidence (e.g., heartbeat cadence and stop reason).
- The exact local replacement validation command and pass status.
- Why the local command supersedes the sparse worker output for this route.

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
