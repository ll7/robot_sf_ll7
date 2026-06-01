---
name: goal-issue-discovery
description: Use for an autonomous Robot SF issue-discovery loop that finds bounded improvement opportunities
  and creates evidence-graded GitHub issues; not for implementation.
category: github-issue
kind: orchestrator
phase: analysis
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to:
- gh-issue-creator
- gh-issue-sequencer
- gh-issue-priority-assessor
- agentic-eval
- auto-improvement
- autoresearch
- context-map
output_schema: skill_run_summary.v1
aliases:
- issue-discovery
---

# Goal Issue Discovery

Use this skill when the user wants a bounded, low-noise issue discovery pass.

This skill orchestrates:
- `gh-issue-creator`
- `gh-issue-sequencer`
- `gh-issue-priority-assessor`
- `agentic-eval`
- `auto-improvement`
- `autoresearch`
- `context-map`

It does not implement issues. It defines scope, evidence quality, duplication control, and stop
conditions.

## Trigger Boundary

Use this skill when the user asks to discover, scan, audit, or collect improvement opportunities and
turn them into GitHub issues.

Do not use it for:
- implementing an existing issue,
- reviewing or fixing a PR,
- broad research synthesis without issue creation,
- one-off issue drafting when `gh-issue-creator` is sufficient.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `docs/context/goal_driven_agent_loops_2026-05-13.md`
- `docs/context/issue_713_batch_first_issue_workflow.md`
- `.agents/skills/gh-issue-creator/SKILL.md`
- `.agents/skills/gh-issue-sequencer/SKILL.md`
- `.agents/skills/context-map/SKILL.md`
- `scripts/dev/check_skills.py` (for local syntax/shape sanity)
- `scripts/dev/check_skills.py --preflight goal-issue-discovery` (for preflight validation before discovery loop)

## Preflight (required, once)

State these items before discovery:
- Scope: repo, benchmark, planner, docs, tests, performance, or explicit paths.
- Lane policy: one bounded lane per pass unless user requests broad mode.
- Write mode: issue creation + Project #5 updates allowed by default.
- Stop condition set: time budget, blocker, API/auth failure, or user stop.
- Exclusions: user-requested areas or explicit skip files.

Do not request additional confirmation for this preflight.

## State Machine

Each candidate is in exactly one state:
- `candidate`
- `duplicate`
- `needs_more_evidence`
- `issue_ready`
- `issue_created`
- `issue_updated`
- `skipped`

Do not revisit `duplicate` or `skipped` in the same run unless new evidence appears.

## Evidence Grade (required)

Choose one:
- `observed`: backed by direct evidence (command output, failing tests, code path, benchmark output,
  stale doc, TODO, trace).
- `inferred`: likely pattern or gap from structure or repeated behavior.
- `proposal`: valuable idea without immediate executable evidence.

Use `proposal` for ideas only; do not frame as bugs/regressions.

## Candidate Readiness

Open or update only candidates with:
- bounded scope and clear problem/opportunity statement,
- affected area (files, docs, workflow, benchmark),
- evidence grade,
- falsification plan (how to prove it wrong),
- explicit non-goals,
- definition of done,
- estimated value or risk.

Split broad ideas before writing.

## Workflow

1. Establish lane and scan surface.
2. Collect candidates from repo-native sources (TODOs, failed or disabled tests, context notes, issue
   gaps, benchmark notes, docs staleness, API friction points).
3. For each candidate:
   - classify state,
   - set evidence grade,
   - record confidence (high if direct signal, medium if one-step derived, low if speculative).
4. De-duplicate against open/closed issues before creation.
5. Draft/update through `gh-issue-creator` only when state is `issue_ready`.
6. Route Project #5 metadata in a single batch pass with score sync once at the end.
7. Stop on no new candidates, budget/time expiry, or blocked writes.

## Proof, Validation, and Artifact Rules

- Do not use local `output/` contents as proof unless tracked or durably referenced.
- Do not count fallback/degraded/fail-open benchmark runs as successful proof.
- Benchmark/planner candidates must name scenario set, seeds/source evidence, and expected
  reproduction command.
- Any paper-facing statement from discovery evidence is forbidden.

## Anti-Loop and Race Conditions

- Stop revisiting candidates already marked `issue_created`, `issue_updated`, `duplicate`, or `skipped`
  unless new evidence arrives.
- Before creating an issue, re-check open/closed state to avoid duplicate creation races.
- Use a single lane by default.

If API/project writes fail repeatedly, emit a short handoff and stop.

## Delegation Failure Recovery

Each child skill or worker may fail. Handle failures per scenario:

- `gh-issue-creator` failure:
  - If issue creation fails due to duplicate detection, log the duplicate
    and skip. Do not retry.
  - If template render or API write fails, record the error, emit a handoff
    note, and continue to the next candidate.

- `gh-issue-sequencer` failure:
  - If Project #5 ordering is unreachable, skip priority normalization and
    use chronological order. Log the failure.

- `gh-issue-priority-assessor` failure:
  - If priority assessment fails, leave the issue unprioritized and continue.
    Do not block the discovery pass.

- `agentic-eval` or `auto-improvement` or `autoresearch` failure:
  - If the evaluation/experiment tool errors, log the failure and skip the
    candidate. Do not halt the discovery scan.

- `context-map` failure:
  - If the context map cannot be generated, fall back to file listing and
    grep-based scanning. Log the degraded mode.

- General environment failure (auth, disk, network):
  - Stop the discovery loop and report the blocker with the failing command,
    exit code, and minimal next action.

Do not retry a child skill on the same candidate if it failed twice with the
same error. Record the recovery action and continue.

When a delegated worker produces a reusable workflow lesson, include an
`agent_run_self_review.v1` companion summary.

For benchmark or planner issues, use `review-benchmark-change` semantics before claim-like wording.
- Do not make paper-facing claims from discovery evidence alone.

## Output Requirements

- discovery scope and stop condition,
- candidates seen by state and confidence,
- created/updated issues with grade + evidence,
- duplicates intentionally avoided,
- Project #5 writes + score-sync status,
- deferred/blocked candidates and race failures.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.


## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.
