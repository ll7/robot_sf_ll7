---
name: pr-hindsight-review
description: Review merged PRs after fact to decide whether autonomous routing produced useful progress or readiness-treadmill work.
category: github-pr
kind: analysis
phase: analysis
requires_write: false
requires_benchmark_artifacts: false
requires_slurm: false
delegates_to: []
output_schema: skill_run_summary.v1
aliases:
  - pr-retrospective
---

# PR Hindsight Review

## When to use
Use this skill after PR merges when a maintainer needs a routing retrospective:
- whether a PR completed its linked issue slice,
- whether it produced useful evidence progress,
- whether a bounded successor routing step is needed.

Use this for autonomous routing retrospectives where a route may have drifted toward readiness-treadmill work.

## Default mode
Default behavior is **read-only**.

Inspect issues, PRs, branches, commits, labels, validation evidence, and changed files.

Do not close issues, comment on PRs, edit labels, edit queue/Project metadata, create successor issues, merge, delete, release, or submit compute work without explicit authorization.

## Required packet fields
- `pr`: PR number URL.
- `title`: PR title.
- `state`: `merged`, `closed-unmerged`, or `open`.
- `merged_at`: merge timestamp when available.
- `linked_issues`: linked issue numbers (and issue references found in PR body, branch, or commits).
- `route_source`: why the PR was selected (ready-queue rank, maintainer request, review follow-up, opportunistic cleanup).
- `scope_summary`: smallest factual scope completed by the PR.
- `successor_judgment`: `no_successor_needed`, `needs_successor`, or `existing_parent_issue_remains_open`.
- `routing_value`: dissertation or evidence value, with conservative caveats.
- `routing_cost`: validation/review/queue/engineering cost observed in hindsight.
- `routing_lesson`: queue heuristic to preserve or change.
- `follow_up_action`: `no_action`, `keep_parent_issue_open`, `draft_successor_packet`, or `maintainer_decision_needed`.
- `verdict`: one of the verdict taxonomy values.
- `confidence`: `low`, `medium`, `high`, with condition that would change judgment.

## Verdict taxonomy
- `complete_progress`: PR fully completed the linked issue slice with no bounded successor needed.
- `partial_existing_parent`: PR made useful partial progress; remaining work is already tracked by an open parent issue.
- `needs_successor`: PR made useful partial progress and remaining work is not already tracked in an open parent.
- `duplicate_or_redundant`: PR overlaps existing coverage without durable proof, contract, artifact, or decision gain.
- `readiness_treadmill`: PR mainly consumed validation/readiness effort without meaningful research/evidence boundary movement.
- `blocked_or_inconclusive`: Packet lacks enough evidence to classify progress without more inspection.

## Workflow
1. Confirm read-only mode is active (unless explicit mutation authorization is provided).
2. Gather PR packet inputs: body, changed files, commits, linked references, validation evidence, and explicit exclusions.
3. Check linked issue state only as needed for full/partial/duplicate/successor decisions.
4. Emit one verdict and one successor judgment per PR.
5. For partial outcomes, name the bounded remainder explicitly.
6. If parent issue remains open, prefer `partial_existing_parent` over `needs_successor`.
7. Return a compact table and packet fields.

## Guardrails
- Do not treat merge status or green checks as full proof of issue completion.
- Do not treat fallback/degraded benchmark runs as evidence of progress.
- Do not promote diagnostic-only outcomes into paper, dissertation, benchmark, or model-provenance claims.
- Do not create successor issues by default; recommend draft packets only.
- Do not close parent issues or edit queue metadata during hindsight review.
- Keep confidence and caveats explicit when evidence is incomplete.

## Output
Return output compatible with `skill_run_summary.v1`.

## Tracked first-batch note
 - [`docs/context/pr_hindsight_review_first_batch_2026-06-30.md`](../../../docs/context/pr_hindsight_review_first_batch_2026-06-30.md).
