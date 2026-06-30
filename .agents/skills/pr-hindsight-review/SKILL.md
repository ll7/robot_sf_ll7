---
name: pr-hindsight-review
description: Review merged PRs to decide whether autonomous routing produced useful progress versus readiness-treadmill work.
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
Use this skill when maintainers need a retrospective packet for merged PRs and routing outcomes.

- whether the PR delivered the linked issue slice
- whether it produced evidence useful for dissertation/benchmark progress
- whether a bounded routing follow-up is needed

## Default mode
Default behavior is **read-only**.

Inspect issues, PRs, branches, commits, labels, validation evidence, and changed files.

Do not close issues, comment on PRs, edit labels, edit queue/Project metadata, create successor issues, merge, delete, release, or submit compute work without explicit authorization.

## Required packet fields
- `pr`: PR number and URL.
- `title`: PR title.
- `state`: `merged`, `closed-unmerged`, or `open`.
- `merged_at`: merge timestamp when available.
- `linked_issues`: linked issue numbers found in PR body/branch/commits.
- `route_source`: why PR was selected (ready-queue rank, maintainer request, review follow-up, opportunistic cleanup).
- `scope_summary`: smallest factual scope completed by the PR.
- `successor_judgment`: `no_successor_needed`, `needs_successor`, `existing_parent_issue_remains_open`.
- `routing_value`: dissertation/research evidence value, with explicit caveats.
- `routing_cost`: validation, review, queue, or engineering cost observed in hindsight.
- `routing_lesson`: queue heuristic to preserve or change.
- `follow_up_action`: `no_action`, `keep_parent_issue_open`, or `document_drift`.
- `verdict`: one value from the verdict taxonomy below.

## Verdict taxonomy
- `complete_progress`: PR completed the identified slice with useful evidence or contract closure and no obvious gaps.
- `partial_existing_parent`: PR made useful progress while bounded remainder is already tracked in an open parent issue.
- `needs_successor`: PR made useful partial progress and remaining work is not clearly tracked elsewhere.
- `duplicate_or_redundant`: PR duplicated existing coverage without durable new evidence or routing value.
- `readiness_treadmill`: PR mostly consumed readiness or validation effort without meaningful research/progress movement.
- `blocked_or_inconclusive`: available evidence was insufficient for confident progress classification.

## Workflow
1. Confirm read-only mode is active unless explicit authorization overrides are present.
2. Gather packet inputs: PR body, changed files, commits, linked references, and validation evidence.
3. Check linked issue state only as needed for full/partial/duplicate/successor decisions.
4. Emit one verdict and one successor judgment per PR.
5. For partial outcomes, name bounded remainders explicitly.
6. If a parent issue remains open, prefer `partial_existing_parent` over `needs_successor`.
7. Return one compact packet per PR.

## Guardrails
- Do not treat merge status or green checks as full proof of issue completion.
- Do not treat fallback or degraded benchmark runs as routing progress.
- Do not promote diagnostic-only outcomes into paper, dissertation, benchmark, or model-provenance claims.
- Do not create successor issues by default; recommend draft packets only.
- Do not close parent issues or edit queue metadata during hindsight review.
- Keep confidence caveats explicit when evidence is incomplete.

## Output
Return output compatible with `skill_run_summary.v1`.

## Tracked first-batch note
- [docs/context/pr_hindsight_review_first_batch_2026-06-30.md](../../../docs/context/pr_hindsight_review_first_batch_2026-06-30.md)
