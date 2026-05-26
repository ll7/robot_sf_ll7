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

## Preflight

Record at start:
- Issue source: queue filter, explicit list, Project #5 lane, or open-issues sweep.
- Write permissions: branch/commit/PR/project writes allowed by default.
- Stop condition: queue exhausted, time budget reached, ambiguous issue contract, environment/auth blocker,
  validation dead-end, user stop.
- Exclusions: benchmarks blocked by environment, blocked/decision-required issues, external-only work.

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
- implementation can fit in one coherent PR.

If not eligible:
- send to `issue-audit` or `gh-issue-clarifier`,
- or create follow-up issue via `gh-issue-creator`.

## Queue Policy

Default order:
1. Project `#5 Ready`
2. Project `#5 Todo`
3. Project `#5 Tracked`
4. explicitly requested
5. other eligible open issues

Prioritize by:
- clearer contract,
- lower validation cost,
- smaller diff,
- less semantic risk,
- older queue age.

## Process

1. Select one issue (`gh-issue-sequencer` output or explicit user target).
2. Create/checkout isolated implementation branch.
3. Implement only in-scope behavior and required tests/docs.
4. Validate using the narrowest meaningful level first, then expand.
5. If validation fails and failure is fixable, adjust and rerun; otherwise record blocker.
6. Prepare proof and branch handoff via `gh-pr-opener`.
7. Open PR, then report and move to next queue item.

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

## Race-Condition / Multi-Agent Safety

- Operate one implementation branch at a time by default.
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
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.


## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.


## Output

Return the schema named by the `output_schema` frontmatter field, or a compact equivalent when the caller does not require YAML.
