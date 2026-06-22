# Token-Efficient Active Thread Profile

Use this header when starting or resuming a long autonomous thread, delegated
batch, or explicitly token-saving workflow. Keep each value short, point to
artifacts instead of pasting raw logs, and update the profile when scope or proof
changes.

```markdown
## Active Profile

- goal: [one sentence objective]
- task_class: [docs | workflow | runtime-code | benchmark | metric | schema | model-provenance | paper-facing | mixed]
- validation_tier: [cheap-docs | skill-sync | focused-tests | full-readiness | benchmark-proof]
- scope: [files, issues, PRs, or batch included]
- out_of_scope: [nearby work deliberately deferred]
- worktree: [absolute or repo-relative worktree path and branch]
- context_budget: [compact snapshots first; raw logs only by artifact path]
- resume_checkpoint: [active ledger path plus loaded skills/docs, current PR/issue/head SHA, and next stale-state trigger]
- route_cache: [unavailable routes and reset times, known-good worker lane, and current CI monitor command]
- delegation_artifacts: [RESULT.md, changed files, diffstat, validation, blockers, mutations]
- output_budget: [parent-thread summary length and exact artifact paths to return]
- stop_guard: [usage threshold, wall-clock limit, or external blocker]
- validation_plan: [commands or checks that prove this task class]
- handoff_target: [PR, issue comment, context note, or final handoff]
```

## Field Rules

- `task_class` names the dominant risk surface. Use `mixed` only when one
  thread intentionally combines more than one class, then name the strongest
  validation tier that applies.
- `validation_tier` follows the proportional readiness matrix in
  `AGENTS.md` and `docs/maintainer_values.md`. Use
  `docs/context/issue_1512_issue_archetypes.md` for issue archetype language
  instead of redefining classes here.
- `context_budget` defaults to compact helpers before broad reads:
  `uv run python scripts/dev/autopilot_state_snapshot.py --include-worktrees`,
  `uv run python scripts/dev/snapshot_issue_batch.py --json`,
  `uv run python scripts/dev/snapshot_pr_queue.py --json`, and
  `uv run python scripts/dev/run_compact_validation.py -- <command>`.
- `resume_checkpoint` should be the first thing refreshed after compaction or
  automatic continuation. If it is fresh, do not reread full skills, broad
  issue queues, or full worktree inventories before the next mutation.
- `route_cache` keeps quota and invocation facts close to the active work:
  Spark usage-limit resets, failed helper flags, the exact bounded CI monitor
  command, and the current fallback worker. Reuse it before retrying a route.
- `delegation_artifacts` are route evidence, not task acceptance. Codex still
  inspects the diff, verifies changed files, and runs the selected validation.
- `output_budget` should return the result, changed files, validation status,
  artifact paths, accepted/rejected/rerouted delegates, risks, follow-ups, and
  current usage. Raw logs stay in the common Git-dir agent-run artifacts unless
  the failure itself needs a short excerpt.
- `stop_guard` is binding for autonomous loops. When a usage threshold fires,
  record the pause in the common Git dir and avoid starting a fresh batch.

## Delegated Worker Artifact Contract

Each delegated implementation or review worker should return a compact artifact
set, preferably in a `RESULT.md` or equivalent final bundle:

- files inspected and files changed;
- diffstat or summary of edits by file;
- validation commands run, exit status, and artifact paths;
- blockers, uncertainty, and deferred follow-ups;
- remote-visible mutations performed, or `none`;
- recommendation: `accept`, `reject`, or `reroute`.

Workers should not create labels, comments, PRs, merges, or other GitHub
mutations unless the parent explicitly delegates that mutation. Parent Codex
acceptance requires local diff review plus the validation tier named in the
active profile.
