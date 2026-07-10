---
name: goal-issue-implementation
description: Use for an autonomous Robot SF issue-to-PR loop that selects eligible GitHub issues, implements
  one scoped issue at a time, validates, pushes, and opens PRs.
category: github-issue
kind: orchestrator
phase: implementation
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to:
- gh-issue-sequencer
- gh-issue-autopilot
- implementation-verification
- pr-ready-check
- gh-pr-opener
- gh-issue-creator
- context-note-maintainer
- issue-splitter
output_schema: issue_to_pr_summary.v1
aliases:
- issue-queue-runner
---

# Goal Issue Implementation

Use this skill when the user asks for goal-driven implementation of open issues.

It is an orchestrator over:

- `gh-issue-sequencer`
- `gh-issue-autopilot`
- `implementation-verification`
- `pr-ready-check`
- `gh-pr-opener`
- `gh-issue-creator`
- `context-note-maintainer`
- `issue-splitter`

It does not define subordinate command details; it standardizes queue policy, evidence and proof
requirements, and loop boundaries.

## Trigger Boundary

Use this skill when the user asks to implement open issues through branches, validation, and PRs.

Do not use it for:
- ambiguous issues that need clarification before coding,
- discovering new work without implementation,
- reviewing existing PRs,
- merging PRs or rewriting contributor history.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `docs/code_review.md`
- `docs/context/goal_driven_agent_loops_2026-05-13.md`
- `docs/context/issue_713_batch_first_issue_workflow.md`
- `.agents/skills/implementation-verification/SKILL.md`
- `.agents/skills/pr-ready-check/SKILL.md`
- `.agents/skills/gh-pr-opener/SKILL.md`
- `.agents/skills/context-note-maintainer/SKILL.md`
- `.github/PULL_REQUEST_TEMPLATE/pr_default.md`
- `scripts/dev/check_skills.py --preflight goal-issue-implementation` (for preflight validation before implementation loop)

## Preflight

Record at start:
- Issue source: queue filter, explicit list, Project #5 lane, or open-issues sweep.
- Write permissions: branch/commit/PR/project writes allowed by default.
- Stop condition: queue exhausted, time budget reached, ambiguous issue contract, environment/auth blocker,
  validation dead-end, user stop.
- Exclusions: benchmarks blocked by environment, blocked/decision-required issues, external-only work.
- Instruction precedence: current maintainer direction and `docs/maintainer_values.md` override stale
  workflow prose. Treat Project #5 ordering as advisory when it conflicts with fresh maintainer
  direction or evidence; record the override and defer Project metadata cleanup when quota or API
  limits make it impractical.
- Autonomy default: for bounded workflow cleanup, proceed without routine confirmation when
  assumptions, uncertainty, evidence grade, and follow-up risks are labeled in the issue, PR, or
  handoff.

Do not ask for extra confirmation after this preflight.

## State Machine

Each issue is in exactly one state during the loop:
- `queued`
- `ineligible`
- `selected`
- `implementing`
- `validating`
- `blocked`
- `pr_opened`
- `deferred_followup`
- `skipped`

Allowed transitions:
`queued -> selected -> implementing -> validating -> pr_opened` with terminal `blocked`, `deferred_followup`,
or `skipped` exits.

Do not revisit `blocked` or `skipped` issues in the same goal run unless one of these changed:
- issue body/labels,
- linked PR state,
- required environment,
- linked issue policy/baseline.

## Eligibility

Process only when all are true:
- issue is open and not labeled blocked/invalid/duplicate/decision-required,
- scope is clear and bounded,
- acceptance criteria are inferable without changing intent,
- proof path is available or has a documented fallback that preserves contract,
- no open PR already covers the issue,
- any issue body/comment `source_pr`, linked PR, `Closes`/`Refs`, or explicit prerequisite PR is
  merged or otherwise available on current `origin/main` when the run must start from clean main,
- implementation can fit in one coherent PR.

If not eligible:
- send to `issue-audit` or `gh-issue-clarifier`,
- or create follow-up issue via `gh-issue-creator`.

Before selecting or branching for a ready issue, inspect the issue body and recent comments for
`source_pr`, linked PRs, `Closes`/`Refs`, and explicit prerequisite PRs. If an open PR already
covers the issue, classify it as `covered_by_pr`, route it to `state:running`, and do not
reimplement. If the issue is a follow-up that depends on an unmerged source PR, classify it
blocked/unavailable for clean-main work with the unblock condition "source PR merged to
`origin/main`", unless the user explicitly chooses a stacked-PR route.

For token-efficient orientation, collect a compact snapshot before broad parent reads:

```bash
uv run python scripts/dev/autopilot_state_snapshot.py \
  --include-worktrees \
  --claim-issue <issue-number> \
  --issue-search "is:issue is:open <issue-number>"
```

Use the snapshot to seed worker prompts with issue/claim/worktree freshness and candidate PR
headline state, but do not treat it as authority for final branch, claim, publication, or merge
decisions. If `ok: false`, a claim is stale against `origin/main`, or the needed field is missing,
run the specific raw `gh`/`git` command named by the snapshot source metadata.

### Delegated Implementation Artifacts

For delegated implementation review and execution workers, enforce artifact-first evidence:

- Workers must write, at minimum, `result.json`, `RESULT.md`, `diffstat.txt`, and `validation.json`
  under a compact artifact directory.
- The parent must read these artifacts first, in that order.
- Parent review flow after delegation:
  1. Inspect `result.json` (status, failures, command list, suspicious signals).
  2. Inspect `RESULT.md` (decision and rationale summary).
  3. Inspect `diffstat.txt`.
  4. Run targeted local verification and diff reads for the reported files/commits before opening logs.
- Raw logs are read only when artifacts are missing, inconsistent, failed, or suspicious.
- Worker prompts must cap parent-thread raw output at about 200 lines. Require `rg -l`, `rg --files`,
  and bounded `sed -n` for search/read tasks; forbid broad `rg -n .` and full file reads unless a
  context pack or compact snapshot failed. Larger output must go to private artifacts with only a
  compact summary returned to the parent.
- Treat worker wrapper completion, worker `route_status: complete`, or any candidate commit as route
  evidence only. Issue implementation is complete only after local revalidation passes and cleanup checks
  are updated.

## Queue Policy

Default order:
1. Project `#5 Ready`
2. Project `#5 Todo`
3. Project `#5 Tracked`
4. explicitly requested
5. other eligible open issues

The Project #5 order this reads is leverage-aware: `goal-autopilot`'s `prioritize` phase auto-fills
**empty** priorities via `gh-issue-priority-assessor` (`--only-empty`) before this phase runs, so
research leverage (claim-boundary/hypothesis → `Improvement`; headline-companion/unblocks →
`Unlock Factor`; local-vs-gated → `Success Probability`) is already encoded in the score. When
running this skill standalone (no autopilot) and an eligible candidate has an **empty** priority,
auto-fill it first with the same `--only-empty` pass so ranking is not driven by unscored gaps; never
overwrite an existing priority to win a tie.

Prioritize by (tie-breakers within the leverage-scored order):
- clearer contract,
- lower validation cost,
- smaller diff,
- less semantic risk,
- older queue age.

When the user asks for research progress rather than generic issue throughput, treat this as
adapted guidance: prefer issues that should close or revise a hypothesis, move a claim boundary,
record a useful negative result, synthesize accumulated diagnostics, or unblock a durable
experiment. Use the compact research-result contract in
`docs/context/goal_driven_agent_loops_2026-05-13.md` when drafting the issue/PR proof surface. Do
not treat this as a hard eligibility rule; support issues may still be the right next step when they
remove a concrete blocker. If the remaining queue is mostly docs cleanup or another diagnostic
extension under an already-expanded research parent, propose a synthesis issue or synthesis pass
before adding more exploratory children.

When re-ranking research backlog issues, do not close an interesting optional path simply because it
is not the best next issue. Leave it open at lower priority with a short reason and revival
condition. Close only duplicate, invalid, or fully superseded issues, or issues that no longer
preserve useful research optionality. Splitting or synthesis is preferred when a broad issue still
contains one implementable child or a reusable research question.

## Queue Exhaustion Audit

Before declaring the implementation queue exhausted, first run the closed-issue state-label hygiene
guard:

```bash
uv run python scripts/dev/closed_state_label_hygiene.py
```

This guard is read-only, avoids Project #5 writes, and exits non-zero with a
`closed_state_label_hygiene.v1` JSON report when any closed issue still carries `state:ready`,
`state:running`, or `state:blocked`. Treat failures as stale queue metadata to clean up or report
before claiming that the active implementation queue is exhausted.

Then run one final read-only implementability audit over:
- open issues labeled `state:ready`,
- open issues that lack any `state:*` label.

If these local filters appear exhausted or dominated by blocked/SLURM-only work, run one bundled
broad queue scout before declaring the queue empty. Prefer a cheap local or Qwen scan plus one
substantial Copilot pass when available; treat scout output as route evidence only until the main
agent verifies it locally. Before selecting, claiming, branching for, or reporting a scout-proposed
issue as ready, run the canonical complete-thread read (issue #5148):

```bash
uv run python scripts/dev/gh_issue_rest.py thread <number> --repo ll7/robot_sf_ll7
```

This tries the concise `gh issue view --comments` path and falls back to paginated REST only for
the known `repository.issue.projectCards` GraphQL failure that breaks `gh issue view --comments`
and `gh issue view --json ...comments` on some GitHub CLI versions. For structured fields, use
`uv run python scripts/dev/gh_issue_rest.py view <number> --json number title state url labels
comments` (REST-only, normalized). Plain `gh issue view <number> --repo ll7/robot_sf_ll7 --json
number,title,state,labels,body,comments,url` is NOT a reliable first-step read because it requests
the deprecated classic-Projects field and exits before returning content on affected hosts; treat
it as an equivalent REST read only when you have already confirmed it works on the host. Confirm
the repository
owner/name, open state, labels, body, recent comments, linked/covering PRs, and ready eligibility
against current GitHub state. Known scout failure modes include stale open/closed/blocked state,
wrong repo-owner URLs, missing recent comments, and duplicate PR coverage; do not acquire
`agent-claims/issue-<number>` or create a branch from scout text alone.
When live labels conflict with recently merged dependency PRs, closeout comments, or issue links,
perform a stale-blocker recheck before skipping the issue. Route one scout specifically to answer
whether the blocker is still true, then confirm the answer with REST `gh api` or local `git`
evidence before changing labels or selecting the next issue.

The audit should classify whether each remaining issue is actually implementable on the current
machine and with the available durable artifacts. If a supposedly ready issue needs unavailable
hardware, SLURM, CARLA, private artifacts, checkpoint aliases, datasets, or a clearer proof path,
mark it blocked or send it to issue clarification instead of counting the queue as empty. Keep this
audit read-only until the orchestrator has reviewed the proposed label/body changes.

Emit the compact `queue_audit.v1` shape alongside the prose handoff when the queue is exhausted or
nearly exhausted. Each row must include:

- issue number,
- classification,
- `implementable_now`,
- recommended action.

Allowed classifications:

- `parent_or_epic`: parent, epic, decision, or umbrella issue that should produce a child issue,
- `analysis_only`: synthesis, interpretation, or research-only work that is not an implementation
  PR unless the run explicitly targets synthesis,
- `lower_priority_research`: interesting research path that should remain open with a recorded
  deprioritization reason and revival condition,
- `blocked_external`: requires unavailable datasets, private artifacts, external services, CARLA,
  or non-local credentials,
- `blocked_slurm`: requires SLURM/Auxme or another unavailable execution environment,
- `covered_by_pr`: already has an open PR or merged change covering the scope,
- `blocked_other`: blocked for a reason not covered above, with rationale; use for follow-up work
  that depends on an unmerged source PR when the run requires clean-main branching,
- `ready_local`: clear, bounded, locally implementable issue,
- `ambiguous`: needs one clarification before implementation routing,
- `too_broad`: mixes multiple independently validatable PRs,

Parent, epic, and analysis-only issues must not be reported as ready implementation work unless the
run explicitly targets clarification, splitting, or synthesis. Blocked-external and blocked-SLURM
issues must route to `mark_blocked`, `clarify`, or `skip` until the missing artifact, service,
credential, or execution environment is actually available.

If the final audit leaves only parent, epic, decision, or research issues that are not directly
implementable, hand exactly one parent to `issue-splitter` instead of stopping with a prose-only
report. The splitter should produce or create one `Next Implementable Child` only after duplicate
checks show that no equivalent child already exists. Include the companion
`issue_split_summary.v1` shape when issue splitting determines the next route.

Example compact exhausted-queue audit:

```text
Queue exhaustion audit
- Query used:
  gh issue list --state open --label state:ready --json number,title,labels,url --limit 100
  gh issue list --search "repo:ll7/robot_sf_ll7 is:issue is:open -label:state:ready -label:state:blocked -label:state:hold" --json number,title,labels,url --limit 100
- Remaining ready issues:
  - #1234 blocked locally: needs SLURM/Auxme allocation; mark `state:blocked` with unblock condition.
  - #1235 ambiguous: acceptance criteria mix benchmark claim and exploratory probe; route to
    `gh-issue-clarifier`.
  - #1236 too broad for one PR: split into fixture migration, docs migration, and compatibility
    validation issues.
- Remaining open issues without `state:*` labels:
  - 17 proposal/research issues need template repair before implementation routing.
- Best issue-splitting candidate:
  - #1236, because the child issues can have independent validation gates and avoid one broad
    path-rewrite PR.
- Writes applied:
  - none yet; audit is read-only pending orchestrator review.
- Next action:
  - clarify #1235 or split #1236 before claiming the implementation queue is exhausted.
```

Companion `queue_audit.v1` example:

```yaml
queue_audit:
  schema: queue_audit.v1
  query_used:
    - gh issue list --state open --label state:ready --json number,title,labels,url --limit 100
    - gh issue list --search "repo:ll7/robot_sf_ll7 is:issue is:open -label:state:ready -label:state:blocked -label:state:hold" --json number,title,labels,url --limit 100
  issues:
    - issue: "#1234"
      classification: blocked_slurm
      implementable_now: false
      recommended_action: mark_blocked
      rationale: needs SLURM/Auxme allocation
    - issue: "#1235"
      classification: ambiguous
      implementable_now: false
      recommended_action: clarify
      rationale: acceptance criteria mix benchmark claim and exploratory probe
    - issue: "#1236"
      classification: too_broad
      implementable_now: false
      recommended_action: split_parent
      rationale: path rewrite should split into fixture migration, docs migration, and validation
    - issue: "#1237"
      classification: covered_by_pr
      implementable_now: false
      recommended_action: wait_for_pr
      rationale: open PR #1238 covers the scope
    - issue: "#1238"
      classification: lower_priority_research
      implementable_now: false
      recommended_action: keep_open_lower_priority
      rationale: interesting optional path, but current evidence makes synthesis or another issue a better next step
    - issue: "#1239"
      classification: ready_local
      implementable_now: true
      recommended_action: implement
      rationale: bounded docs/tooling change with local validation path
    - issue: "#1240"
      classification: parent_or_epic
      implementable_now: false
      recommended_action: split_parent
      rationale: umbrella issue needs a next implementable child
    - issue: "#1241"
      classification: analysis_only
      implementable_now: false
      recommended_action: synthesize
      rationale: asks for evidence interpretation, not code or docs implementation
    - issue: "#1242"
      classification: blocked_external
      implementable_now: false
      recommended_action: mark_blocked
      rationale: depends on unavailable licensed dataset assets
  best_next_action: implement #1239, or split #1236 if no ready_local row remains
```

Use the prose summary for human readability and `queue_audit.v1` for repeatable comparison across
runs. If a remaining issue only needs a clearer contract, route it to issue clarification. If a
remaining issue bundles several independently validatable changes, create child issues with
`gh-issue-creator` and leave the parent as the coordination issue instead of treating the bundle as
unimplementable.

Route remaining issues by their blocker:
- Use `gh-issue-clarifier` when the issue intent, proof path, acceptance criteria, or ownership is
  unclear.
- Use `gh-issue-creator` or an explicit issue-splitting pass when one ready issue mixes multiple
  independently reviewable PRs.
- Mark the run `blocked` instead of exhausted when the next issue requires unavailable hardware,
  private artifacts, credentials, live external services, or unapproved Project writes.

## Process

1. Select one issue (`gh-issue-sequencer` output or explicit user target).
2. Re-check issue body/comments and open PRs for source-PR dependencies, active coverage, and
   duplicate branch/PR risk before branching.
3. Acquire the cross-machine issue claim before branching:

   ```bash
   uv run python scripts/dev/issue_claim.py acquire <issue-number>
   ```

   The helper creates `agent-claims/issue-<issue-number>` through GitHub's create-ref API, which
   fails when the ref already exists. If the command fails, treat the issue as already claimed by
   another PC or agent, run `uv run python scripts/dev/issue_claim.py status <issue-number>` for the
   handoff, and skip to the next candidate unless the claim is explicitly confirmed stale and
   released.
4. Make the successful claim visible in the issue/project surfaces: move the issue to `In progress`
   or `state:running`, assign the actor when practical, and add a concise issue comment with the
   claim ref, machine/thread, planned branch, and stale-claim cleanup condition.
5. Use detached latest-main checkouts only for read-only discovery, duplicate checks, and GitHub
   issue creation or update work. Before editing docs or code, running PR validation, pushing, or
   publishing, always run the implementation task inside a linked git worktree so the execution/validation
   remains fully isolated.
6. Create or update the active delegation ledger described in
   `.agents/skills/goal-autopilot/SKILL.md` with the claim ref/status, issue, branch/worktree,
   owned files or modules, route/run IDs, validation plan, cleanup status, and next action. Update
   it after every delegated worker/sub-agent start/completion/failure, before any CI wait or compaction-prone
   pause, and after PR creation and claim release. Record route success separately from task
   success.
7. Set up the linked git worktree for the selected issue.
   - Follow `AGENTS.md` "Fresh Worktree Bootstrap" for location, naming, machine-context symlink, and early `origin/main` freshness.
   - Run the bootstrap commands inside the worktree (e.g. `uv sync --all-extras`, activate virtualenv).
8. Delegate the actual implementation to a sub-agent.
   - Define a specialized sub-agent (e.g., inheriting the parent's tools and system prompt) with the workspace configured to share or inherit the worktree directory.
   - Invoke the sub-agent and pass the issue details, the target worktree path, and the validation requirements.
   - The sub-agent must perform planning, execution, and local validation (lint/format/tests) strictly inside that worktree.
   - The sub-agent must generate the required worker artifacts: `result.json`, `RESULT.md`, `diffstat.txt`, and `validation.json` under the artifact directory, and notify the orchestrator when finished.
9. Parent monitors and audits the sub-agent's work:
   - Check the sub-agent's status and wait for it to complete.
   - Once complete, inspect `result.json`, `RESULT.md`, `diffstat.txt`, and run targeted local verification before accepting.
   - If validation or proof is insufficient, instruct the sub-agent to repair it, or mark the issue blocked.
10. Commit/push the completed changes from the worktree and prepare the PR handoff using `gh-pr-opener`.
11. Open the PR and release the transient claim with:
    ```bash
    uv run python scripts/dev/issue_claim.py release <issue-number>
    ```
    Ensure the PR visibly covers the issue before releasing the claim.
12. Tear down the worktree:
    - Follow `AGENTS.md` "Worktree Teardown And Preservation" to clean up the linked worktree and prune references.
    - Move to the next queue item.

Never run unrelated refactors or paper-facing claims in this loop.

## Validation Tiers

Use the minimum required tier for changed surfaces:

- Tier 0: documentation-only and metadata-only changes, small mechanical code.
- Tier 1: CLI/runtime changes, interfaces, shared utility wiring.
- Tier 2: benchmark mode, metric, or planner-sensitive behavior.
- Tier 3: campaign-level or statistical evidence changes.

Before PR creation, rerun freshness gate after latest `origin/main` sync.
Do not use stale validation as proof.

## Proof and Artifact Rules

- Do not count benchmark fallback/degraded execution as success unless task scope explicitly says so.
- Classify all generated outputs before commit:
  - `discard`, `ignored-cache`, `tracked-manifest`, `durable-required`.
- Benchmark artifacts must include command, config, seeds, commit SHA, and provenance before PR handoff.
- If a durable dependency cannot be guaranteed locally, stop with a blocker.

## Confidence

Use one confidence level for final reporting:
- `High`: proof completed and current for branch head.
- `Medium`: evidence exists but depends on external CI or temporary constraints.
- `Low`: required proof/auth/benchmark path unavailable.

Never close an issue with `Low` proof without explicitly marking follow-up work.

## Anti-Loop and Retry

- Do not rerun identical failed validations more than twice without meaningful code/env change.
- If a candidate fails the same gate with no new signal, move to `blocked` and record:
  - failing command,
  - last error,
  - next minimal action.

## Delegation Failure Recovery

Each child skill or worker may fail. Handle failures per scenario:

- `gh-issue-sequencer` failure:
  - If the queue is empty or unreachable, skip queue ordering and fall back to
    explicit user target or open-issues sweep.
  - If Project #5 API writes fail, log the error and continue without priority
    normalization.

- `gh-issue-autopilot` failure:
  - If the issue is ambiguous mid-flow, route to `issue-contract-maintainer`,
    mark the issue `skipped`, and continue to the next candidate.
  - If branch creation fails, record the error and skip the issue.
  - If validation fails twice without meaningful change, mark the issue
    `blocked` and record the failing command and last error.

- `implementation-verification` failure:
  - If evidence is insufficient, record the gaps and move the PR to
    `blocked` instead of `pr_opened`. Do not open a PR with failing claims.

- `pr-ready-check` failure:
  - If the gate fails on fixable issues (lint, format), fix and retry once.
  - If the gate fails on test or coverage, record the failure and move to
    `blocked` instead of `pr_opened`.

- `gh-pr-opener` failure:
  - If the PR already exists for the branch, update the existing PR body
    instead of creating a duplicate.
  - If push fails, record the error and mark the issue blocked.

- `gh-issue-creator` or `issue-splitter` failure:
  - Log the failure, skip the issue, and continue to the next candidate.
  - Do not let a child-creation failure block the implementation queue.

- General environment failure (auth, disk, network):
  - Stop the implementation loop and report the blocker with the failing command,
    exit code, and minimal next action.

Do not retry a child skill on the same issue if it failed with the same error
twice. Record the recovery action and continue.

When a delegated worker produces a reusable workflow lesson, include an
`agent_run_self_review.v1` companion summary.

## Race-Condition / Multi-Agent Safety

- Operate one implementation branch at a time by default.
- Use `uv run python scripts/dev/issue_claim.py acquire <issue-number>` as the first write after
  candidate selection and duplicate-PR checks. The remote `agent-claims/issue-<number>` ref is the atomic
  cross-machine claim; labels, assignments, Project #5 status, and comments are secondary visibility
  signals.
- If acquiring the claim fails, do not branch or implement. Record the existing claim status and
  select another issue. Release only stale or abandoned claims after checking for an open PR or a
  recent issue comment from the claimant.
- Before cleaning stale worktrees at loop end or after a PR handoff, follow `AGENTS.md` "Worktree
  Teardown And Preservation" and record how relevant tracked, untracked, and ignored local changes
  were preserved.
- Before pushing, verify branch state is expected and avoid rewriting branch history.
- If remote branch changed unexpectedly:
  - stop,
  - inspect divergence,
  - avoid force-push,
  - hand off with blocker details.

## Required Output

For each issue completed or stopped, report:
- issue number and eligibility decision,
- current and next state,
- branch name + head SHA,
- validation commands and pass/fail results,
- artifact decision,
- PR URL when opened,
- follow-up issues,
- blocker and next action.

When the queue exhausts, also report the final implementability audit result: remaining ready
issues, remaining open issues missing `state:*`, any labels/body updates applied after review, and
the command or query used to confirm the queue state. Include `queue_audit.v1` rows for exhausted
or nearly exhausted queues so future agents can compare classifications across runs.

When delegated agent or worker behavior produced a reusable workflow lesson, include an
`agent_run_self_review.v1` companion summary or link the inbox note that follows that shape. Do not
promote a lesson into durable skill text unless the evidence is repeated, high-confidence, or
directly explains a costly failure.

## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.


## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.


## Output

Return the schema named by the `output_schema` frontmatter field, or a compact equivalent when the caller does not require YAML.
