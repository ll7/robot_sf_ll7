---
name: goal-issue-discovery
description: "Autonomous goal loop that scans Robot SF for improvement opportunities and opens evidence-graded GitHub issues."
---

# Goal Issue Discovery

## Overview

Use this skill when the user wants an autonomous pass that finds possible bugs, performance gains,
research ideas, feature opportunities, documentation gaps, or maintainability improvements and turns
them into detailed GitHub issues.

This is a goal-driven orchestration skill. It reuses `gh-issue-creator`, `gh-issue-sequencer`,
`gh-issue-priority-assessor`, `context-map`, `agentic-eval`, `auto-improvement`, and
`autoresearch` instead of replacing them.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `docs/context/goal_driven_agent_loops_2026-05-13.md`
- `docs/context/issue_713_batch_first_issue_workflow.md`
- `.agents/skills/gh-issue-creator/SKILL.md`
- `.agents/skills/gh-issue-sequencer/SKILL.md`
- `.agents/skills/context-map/SKILL.md`

## Preflight

At the start of each goal, state:

- discovery scope, such as whole repo, benchmark, planner, docs, tests, or performance,
- write mode: autonomous GitHub issue creation and Project #5 routing are allowed by default,
- stop condition: queue exhausted, explicit time budget reached, auth/API failure, or validation
  blocker,
- exclusions, if the user supplied any.

Do not ask for confirmation after the preflight unless the user explicitly requests a gated run.

## Evidence Grades

Every issue created by this loop must include one evidence grade in the issue body:

- `observed`: backed by a failing command, code path, stale doc, TODO, benchmark output, or other
  directly inspected repo evidence.
- `inferred`: supported by repository structure or repeated patterns, but not yet proven by an
  executable failure.
- `proposal`: a broad feature, research, or improvement idea that is useful enough to track even
  without current failure evidence.

Do not present `proposal` issues as bugs or benchmark regressions.

## Workflow

1. Establish context
   - Inspect `git status --short --branch`.
   - Read local machine guidance if present before expensive checks.
   - Pick one bounded discovery lane per pass unless the user explicitly asks for a broad scan.

2. Search for candidates
   - Prefer repo-native sources: open TODOs, failing or skipped tests, stale docs, complexity or
     timing helpers, benchmark notes, issue-linked context notes, and recent PR/issue patterns.
   - For benchmark or planner ideas, inspect the relevant context notes before making claims.
   - Use Spark subagents only for bounded sidecar discovery, such as locating APIs or summarizing a
     small directory. The main agent owns issue creation and final judgment.

3. De-duplicate before writing
   - Search open and closed issues with `gh issue list` and `gh issue search` when available.
   - If a related issue exists, update or comment on that issue instead of creating a duplicate.
   - If the existing issue is too broad, create a child/follow-up issue and link both directions.

4. Draft issues through the repo template
   - Use `.agents/skills/gh-issue-creator/SKILL.md`.
   - Include: evidence grade, concise goal/problem, scope/non-goals, affected files, validation
     path, risk, estimated value, and definition of done.
   - Use existing labels only unless the discovery itself justifies a new label.
   - Route Project #5 metadata in a separate batch pass.

5. Batch project metadata
   - Follow `docs/context/issue_713_batch_first_issue_workflow.md`.
   - Cache Project #5 IDs once per shell session.
   - Set status and priority conservatively, then run score sync once if score inputs changed.

6. Stop and hand off
   - Stop when no more credible candidates remain, the time budget is reached, or GitHub/project
     writes fail.
   - If stopped early, write a concise handoff note or issue comment listing scanned areas,
     created/updated issues, skipped candidates, and the next recommended lane.

## Guardrails

- Broad ideas are allowed, but issue bodies must make uncertainty explicit.
- Do not rely on local `output/` artifacts as durable evidence unless they are promoted or linked
  through an accepted artifact path.
- Do not classify fallback or degraded benchmark execution as success evidence.
- Do not create large umbrella issues when separate bounded issues would be more executable.

## Output Requirements

Report:

- discovery scope and stop condition,
- areas inspected,
- issues created or updated, with evidence grade,
- duplicates avoided,
- Project #5 writes and score sync status,
- remaining candidate areas or blockers.
