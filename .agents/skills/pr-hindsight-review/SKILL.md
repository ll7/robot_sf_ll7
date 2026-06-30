---
name: pr-hindsight-review
description: Review merged PRs after the fact to decide whether autonomous routing produced useful progress, partial coverage, duplicate coverage, or a successor slice.
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

Use this skill after one or more pull requests (PRs) have merged and the maintainer needs a read-only routing retrospective: what the PR actually completed, what remains, whether the work was dissertation- or evidence-relevant, and whether a bounded successor issue should be recommended.

Use it especially for autonomous routing retrospectives where the question is whether a route created durable research progress or mostly produced readiness treadmill work.

## Default Mode

Default mode is read-only. Inspect GitHub issues, PRs, branches, commits, labels, validation notes, and changed files, but do not close issues, comment on issues or PRs, edit labels, edit queue or Project metadata, create successor issues, merge, release, delete, or submit compute work.

Mutation mode is future and optional. It may create a successor issue only when the caller explicitly requests mutation, the draft successor packet is present, and repository authorization allows GitHub issue writes. Even in mutation mode, keep queue, label, merge, release, delete, and compute-submission actions separate unless explicitly authorized.

## Required Packet Fields

For each reviewed PR, collect and report:

- `pr`: PR number and URL.
- `title`: PR title.
- `state`: `merged`, `closed-unmerged`, or `open`.
- `merged_at`: merge timestamp when available.
- `linked_issues`: issue numbers named by PR body, branch, commits, or explicit references.
- `route_source`: why this PR was selected, such as ready queue rank, maintainer request, review follow-up, or opportunistic cleanup.
- `scope_summary`: smallest factual summary of the merged diff.
- `validation_evidence`: commands, checks, or review evidence recorded by the PR.
- `out_of_scope`: explicit exclusions stated in the PR body or inferred from issue contract.
- `issue_disposition`: whether the PR fully closes, partially covers, duplicates, or leaves a bounded remainder for each linked issue.
- `successor_judgment`: `no_successor_needed`, `needs_successor`, or `existing_parent_issue_remains_open`.
- `routing_value`: dissertation or evidence value, stated conservatively with caveats.
- `routing_cost`: validation, review, queue, or readiness cost noticed in hindsight.
- `routing_lesson`: routing rule or queue heuristic hindsight suggests preserving or changing.
- `follow_up_action`: `no_action`, `keep_parent_issue_open`, `draft_successor_packet`, or `maintainer_decision_needed`.
- `verdict`: one value from the verdict taxonomy.
- `routing_label`: one value from the original routing-label set when a progress-disposition verdict is not enough.
- `confidence`: `low`, `medium`, or `high`, with the condition that would change the judgment.

## Verdict Taxonomy

- `complete_progress`: PR fully satisfies the linked issue slice and no bounded successor is needed.
- `partial_existing_parent`: PR produced useful partial progress, and the remaining work is already tracked by an open parent issue.
- `needs_successor`: PR produced useful partial progress, but the linked parent is completed, closed, or too broad for the remaining bounded slice; recommend a successor issue packet.
- `duplicate_or_redundant`: PR overlaps existing coverage without adding a clear new proof, contract, artifact, or durable decision.
- `readiness_treadmill`: PR mostly spent validation or queue effort without moving a research, evidence, workflow, or maintainability boundary enough to justify similar future routing.
- `blocked_or_inconclusive`: hindsight packet lacks enough evidence to classify progress without more source inspection or maintainer input.

## Original Routing Labels

Issue #3795 requested this controlled routing-label set. Preserve it as a secondary label so hindsight packets can compare routing outcomes consistently:

- `evidence_yielding`: PR created or protected durable evidence, claim-boundary proof, or benchmark/metric contract clarity.
- `useful_readiness`: PR did not itself prove the research claim, but materially reduced risk or review friction for a still-valid next evidence slice.
- `readiness_treadmill`: PR mostly consumed validation or review effort without enough durable progress to justify similar future routing.
- `duplicate_or_overlapping`: PR repeated existing coverage or overlapped another active route without a clear new decision.
- `too_low_priority`: PR may be correct, but hindsight says the route was less valuable than available alternatives.
- `needs_successor`: PR made useful partial progress and should produce a bounded non-duplicate successor slice.

Use `verdict` for issue disposition and `routing_label` for route-quality judgment. For example, a `partial_existing_parent` verdict may still carry `useful_readiness`, while a `complete_progress` verdict may carry `evidence_yielding`.

## Successor Issue Packet

When the verdict or routing label is `needs_successor`, include a draft successor issue packet without posting it by default. The packet must include:

- `title`: proposed successor issue title.
- `parent`: parent issue and PR reference.
- `completed_scope`: what was already completed.
- `remaining_scope`: bounded work not completed by the reviewed PR.
- `dissertation_or_evidence_relevance`: why the remainder still matters.
- `first_implementation_slice`: first bounded Codex implementation slice.
- `validation_expectation`: cheapest proof that would make the successor reviewable.
- `non_duplication_rationale`: why the slice is not duplicate completed coverage.

Queue and scout policy should exclude completed parent issues, but it may route an explicitly named successor issue when the hindsight packet explains the remaining non-duplicate slice.

## Workflow

1. Confirm the caller's authorization and keep the review read-only unless mutation is explicitly authorized.
2. Fetch or inspect each PR packet: PR body, changed files, commits, linked issues, labels, merge state, validation evidence, and explicit exclusions.
3. Inspect linked issue state only as needed to decide full, partial, duplicate, or successor disposition.
4. Classify the PR using the verdict taxonomy, original routing label, and successor judgment.
5. For partial work, name the remaining bounded slice explicitly. If the parent issue remains open, prefer `partial_existing_parent` over a new successor recommendation.
6. Record caveats when the PR body or issue text is incomplete, compressed, or ambiguous.
7. Output a compact retrospective table plus any successor issue packets.

## Guardrails

- Do not treat merged status, passing checks, or `merge-ready` labels as proof that the linked issue was fully satisfied.
- Do not classify fallback or degraded benchmark execution as benchmark-strengthening evidence.
- Do not promote diagnostic-only results into paper, dissertation, benchmark, or claim-map changes.
- Do not create successor issues by default; recommend draft packets only.
- Do not close parent issues or edit queue metadata during hindsight review.
- Do not invent linked issue closure when the PR body says the work is support-only, partial, or leaves existing issues open.
- Keep confidence explicit when issue bodies, PR bodies, or validation logs are unavailable.

## Output

Return a compact `skill_run_summary.v1`-compatible summary with:

- reviewed PR list;
- per-PR verdict, routing label, successor judgment, bounded remainder, confidence, and evidence;
- successor issue packets for every `needs_successor` verdict or routing label;
- residual risks and missing evidence;
- confirmation that default read-only restrictions were preserved.

Tracked example: first-batch autonomous routing hindsight note [`docs/context/pr_hindsight_review_first_batch_2026-06-30.md`](../../../docs/context/pr_hindsight_review_first_batch_2026-06-30.md).
