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
- current_request_guard: [newest user request, whether it supersedes the active ledger action, and parked-work state]
- loaded_context_cache: [skills/docs/snapshots already read this phase, plus the condition that requires rereading]
- route_cache: [unavailable routes and reset times, known-good worker lane, and current CI monitor command]
- delegation_artifacts: [RESULT.md, changed files, diffstat, validation, blockers, mutations]
- output_budget: [parent-thread summary length and exact artifact paths to return]
- stop_guard: [usage threshold, wall-clock limit, or external blocker]
- validation_plan: [commands or checks that prove this task class]
- phase_audit_record: [latest compact token-spend review, route changes, and reusable lesson path]
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
- `current_request_guard` prevents stale-goal continuation after compaction or
  user pivots. Before taking the next action, compare the newest user request
  with the active ledger action. If they differ, write a compact parked-work
  handoff with dirty worktrees, active delegates, PR/issue state, and cleanup
  status, then execute the newest request.
- `loaded_context_cache` prevents compaction recovery from becoming another
  broad-read pass. Record each long skill, issue body, PR snapshot, and context
  note already loaded in the current phase, then reread only when the branch,
  issue/PR head SHA, skill file modification time, or explicit stale-state
  trigger changes.
- `route_cache` keeps quota and invocation facts close to the active work:
  Spark usage-limit resets, failed helper flags, the exact bounded CI monitor
  command, and the current fallback worker. Reuse it before retrying a route.
  If a Codex app subagent spawn fails because the model quota is exhausted,
  record the reset time immediately, close the handle when possible, and do not
  retry that model during the same phase.
- `delegation_artifacts` are route evidence, not task acceptance. Codex still
  inspects the diff, verifies changed files, and runs the selected validation.
- `output_budget` should return the result, changed files, validation status,
  artifact paths, accepted/rejected/rerouted delegates, risks, follow-ups, and
  current usage. Raw logs stay in the common Git-dir agent-run artifacts unless
  the failure itself needs a short excerpt.
- `stop_guard` is binding for autonomous loops. When a usage threshold fires,
  record the pause in the common Git dir and avoid starting a fresh batch.
- `phase_audit_record` keeps the token-spend review outside the parent thread.
  Record the latest usage snapshot, reused context cache, skipped broad reads,
  accepted or rejected delegates, and any candidate lesson note. Link the
  common-Git-dir artifact instead of pasting raw route logs.

## Phase Audit

At each phase boundary, review the active thread for token leaks before starting
new work. Keep the audit to one compact paragraph or checklist, and patch the
workflow only when the savings are reusable.

Start every resumed or user-pivoted phase with a five-minute startup gate:

- `newest_request`: quote or paraphrase the latest user request in one line.
- `parked_work`: list any previous PR, worktree, dirty state, active delegate,
  or running job that is being preserved instead of continued.
- `next_mutation`: name the next branch, PR body edit, label, comment, merge,
  job submission, or issue mutation before doing it.
- `freshness_key`: record the PR/issue head SHA, branch base, queue timestamp,
  usage snapshot, or submit-host proof that makes the next action safe.
- `stop_or_continue`: explicitly choose continue, park, handoff, or stop.

- Start from the current `resume_checkpoint`, `loaded_context_cache`, and
  `phase_audit_record`. Reread long skills, issue bodies, PR snapshots, or
  context notes only when their recorded freshness key changed.
- Re-anchor on the newest user request before continuing a prior goal ledger.
  If the request changed, pause the old batch with a parked-work note instead
  of mixing objectives in the same implementation branch or PR body.
- Run one usage check per phase, record the reset window, and reuse it until a
  cooldown, direct user request, or new phase makes it stale.
- Refresh the issue or PR head SHA before implementation and before review
  repair so closed, merged, or superseded work is detected before editing.
- Treat parent-thread search output as a budgeted surface. Prefer file-name
  discovery (`rg --files | rg <focused-pattern>`), `rg -l`, and bounded
  `sed -n` ranges; redirect broad `rg -n` or multi-directory searches to a
  private artifact and report only paths plus the decision-relevant excerpt.
- For implementation-thread audits, summarize the last relevant time window
  into a common-Git-dir artifact first. Report only record counts, top output
  offenders, failed command families, unclear-instruction themes, and the
  artifact path; do not replay raw session JSONL or old transcript chunks in
  the parent thread.
- Use filtered status helpers for local state:
  `uv run python scripts/dev/worktree_hygiene_snapshot.py --repo-status --filter <branch> --json`
  for branch cleanup, and raw `git worktree list --porcelain` only when the
  helper reports a stale entry or missing detail.
- For Slurm work, prove the owning worktree exists on the submit host before
  submitting. A local worktree preflight is not enough when the private submit
  wrapper reaches the cluster through SSH.
- When SLURM is the user's current priority, classify it in the phase audit as
  `single goal-slurm-experiment (gse) lane`, `capacity-aware batch`, `blocked`,
  or `analysis-only`.
  Read only the relevant private-ops submit instructions for the chosen route,
  then record queue freshness, duplicate checks, submit-host worktree proof, and
  immediate health-check plan before any `sbatch` or remote wrapper call.
- Keep CI and validation loops bounded in the parent thread. Store full output
  in compact-validation artifacts and report only command, exit code, failing
  node IDs, artifact paths, and the next action.
- For CI polling, surface only status changes and terminal conclusions when
  repeated JSON payloads would be noisy. If a workflow is still running, prefer
  job metadata and filtered direct job-log excerpts over `gh run view --log`
  dumps that can fail or flood the parent context.
- If the parent thread must poll CI directly, use one-shot snapshots or redirect
  multi-attempt JSON streams to a private artifact. Repeated unchanged
  `pending` payloads should update the ledger, not the parent transcript.
- If GitHub CI runs repo-wide lint/format gates after a branch merges fresh
  `origin/main`, run the matching repo-wide lightweight gate locally when a
  focused changed-file check would miss unrelated-but-current format drift.
- Check `route_cache` before spawning a delegate. If a model or helper is
  quota-blocked, reuse the recorded reset time and route directly to the next
  eligible worker.
- Require delegated workers to return only files inspected, files changed,
  validation, blockers, and recommendation. Ask for raw logs only after the
  parent identifies a specific uncertainty.
- Reject or reroute worker output that lacks compact artifacts before reading
  full logs. Accept sparse or app-agent-only output only after parent-owned diff
  review and validation prove the result independently.
- After every accepted, rejected, or rerouted delegate, append a one-paragraph
  self-review note when the route created reusable evidence about output size,
  artifact quality, quota pressure, or validation cost.
- Before final handoff or a user-requested pivot, list active app subagents and
  long-running worker processes when the tooling supports it. Close obsolete
  app subagents, record any preserved workers or worktrees, and do not leave
  cleanup implicit behind a PR link.
- For worktree cleanup, classify ignored `output/` contents by top-level
  directory, count, and durable-evidence category before expanding paths. Full
  ignored-output listings belong in private artifacts unless a specific path is
  being preserved, promoted, or explained.
- When usage is close to the stop guard, finish the active PR or blocker record,
  then hand off instead of opening a fresh batch.

## Known Fallbacks

Use this table before retrying failed workflow commands in a parent thread. The
goal is to classify known tool or setup friction quickly, then rerun the exact
bounded fallback once instead of rediscovering it through broad logs.

| Symptom | Preferred fallback | Classification |
| --- | --- | --- |
| `rtk read --range ...` reports an unsupported command shape. | Use `rtk proxy sed -n '<start>,<end>p' <path>` for the same focused file slice. Keep the range bounded and record the failed shape in `route_cache` before retrying. | Tool invocation fallback, not file evidence. |
| A fresh sibling worktree cannot collect focused benchmark tests because an optional dependency such as `torch` is unavailable. | Retry the same focused command through the shared environment wrapper, for example `scripts/dev/run_worktree_shared_venv.sh -- uv run pytest <test-node>`. If the wrapper passes, report the direct command as a worktree setup or optional-dependency issue, not a code regression. | Environment/setup fallback; final PR proof still needs the validation tier named for the change. |
| `scripts/dev/check_pr_followups.py` reports `missing_domain_approval_note` after the Domain-Aware Approval section was filled from template options. | Replace slash-separated option text with concrete comma-separated values, for example `evidence classification, experimental comparison`, or a concrete `NA` reason when approval is not required. Do not relax the approval contract. | PR-body contract repair. |

## Token-Spend Review

Use this compact review before creating workflow-improvement PRs or continuing a
long goal near a usage guard:

- `usage`: latest primary and secondary remaining percentages, reset times, and
  whether the stop guard is close.
- `largest_parent_outputs`: broad commands, full skill reads, raw logs, or
  verbose delegate messages that entered the parent thread.
- `failed_command_patterns`: repeated helper failures, confusing command
  contracts, CI log endpoints that fail while runs are pending, or unclear
  instruction wording that caused retries.
- `delegation_economics`: delegates accepted, rejected, rerouted, or skipped,
  plus whether reviewing them was cheaper than direct Codex work.
- `cache_hits`: skills, docs, snapshots, issue state, route facts, and usage
  readings reused instead of reread.
- `cache_misses`: repeated reads or rediscovery that should become ledger
  fields, route-cache entries, or worker prompt constraints.
- `next_savings`: three to ten concrete changes, each phrased as a rule that
  prevents a repeated token leak and names the evidence that triggered it.
- `parked_work`: prior worktrees, PRs, jobs, delegates, or claims intentionally
  left active because the newest request superseded the old next action.
- `instruction_pr_scope`: exact instruction files to change, validation tier,
  and why the change belongs in reusable guidance instead of one-off handoff
  prose.

## Meta-Workflow Instruction Pull Request Gate

Use this gate when the newest user request asks to improve instructions,
review Codex token spending, audit the last implementation thread, or make the
workflow more reliable. Treat the prior goal loop as parked unless the user
explicitly asks to resume it.

- Start from the `Token-Spend Review` fields above and choose three to ten
  reusable improvements. Each improvement should name the observed leak or
  failure it prevents, such as raw worktree fleet output, full skill rereads,
  stale-goal continuation after a user pivot, helper-path confusion, unbounded
  continuous integration (CI) polling, missing delegate cleanup, or pull
  request (PR) body contract retries.
- Make those improvements acceptance criteria, meaning specific conditions the
  instruction change must satisfy, before editing. If the current instruction
  surface already covers one, do not duplicate it; either tighten the existing
  wording with the new evidence or leave it out of scope.
- Create a fresh docs-or-workflow worktree from `origin/main`. Record parked
  PRs, jobs, dirty worktrees, and active delegates in the ledger, then keep the
  new branch limited to instruction changes.
- Use compact evidence first. For thread history, write summaries to a
  common-Git-dir artifact; for worktrees, use filtered snapshot helpers; for
  skills, reuse the loaded-context cache and reread only the directly edited
  instruction file.
- Validate with the cheapest official path for the changed surface. Docs-only
  changes need diff inspection and referenced-path checks; skill changes also
  need the relevant skill/schema/sync check when available. Full PR readiness
  is reserved for executable or evidence-sensitive changes.
- The Pull Request (PR) body should list the token leaks addressed, files
  changed, validation run, parked work, and any instruction gaps intentionally
  left for a later issue.

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
