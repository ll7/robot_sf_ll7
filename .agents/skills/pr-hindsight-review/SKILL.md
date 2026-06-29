---
name: pr-hindsight-review
description: Review merged PRs after the fact to decide whether autonomous routing produced useful progress, partial coverage, duplicate coverage, or a successor slice.
category: github-pr
kind: analysis
phase: analysis
requires_write: false
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
aliases:
  - pr-retrospective
---
# PR Hindsight Review

## When to use
Use this skill after one or more pull requests (PRs) have merged and the maintainer needs a
read-only routing retrospective: what the PR actually completed, what remains, whether the work was
dissertation- or evidence-relevant, and whether a bounded successor issue should be recommended.

Use it especially for autonomous routing retrospectives where the question is whether a route
created durable research progress or mostly produced readiness treadmill work.

## Default Mode
Default mode is read-only. Inspect GitHub issues, PRs, branches, commits, labels, validation notes,
and changed files, but do not close issues, comment on issues or PRs, edit labels, edit queue or
Project metadata, create successor issues, merge, release, delete, or submit compute work.

Mutation mode is future and optional. It may create a successor issue only when the caller
explicitly requests mutation, the draft successor packet is present, and repository authorization
allows GitHub issue writes. Even in mutation mode, keep queue, label, merge, release, delete, and
compute-submission actions separate unless explicitly authorized.

## Required Packet Fields
For each reviewed PR, collect and report:

- `pr`: PR number and URL.
- `title`: PR title.
- `state`: merged, closed-unmerged, or open.
- `merged_at`: merge timestamp when available.
- `linked_issues`: issue numbers named by PR body, branch, commits, or explicit references.
- `route_source`: why this PR was selected, such as ready queue rank, maintainer request, review
  follow-up, or opportunistic cleanup.
- `scope_summary`: smallest factual summary of the merged diff.
- `validation_evidence`: commands, checks, or review evidence recorded by the PR.
- `out_of_scope`: explicit exclusions stated in the PR body or inferred from issue contract.
- `issue_disposition`: whether the PR fully closes, partially covers, duplicates, or leaves a
  bounded remainder for each linked issue.
- `successor_judgment`: `no_successor_needed`, `needs_successor`, or
  `existing_parent_issue_remains_open`.
- `routing_value`: dissertation or evidence value, stated conservatively with caveats.
- `routing_cost`: validation, review, queue, or readiness cost noticed in hindsight.
- `verdict`: one value from the verdict taxonomy.
- `confidence`: low, medium, or high, with the condition that would change the judgment.

## Verdict Taxonomy
- `complete_progress`: PR fully satisfies the linked issue slice and no bounded successor is needed.
- `partial_existing_parent`: PR produced useful partial progress, and the remaining work is already
  tracked by an open parent issue.
- `needs_successor`: PR produced useful partial progress, but the linked parent is completed,
  closed, or too broad for the remaining bounded slice; recommend a successor issue packet.
- `duplicate_or_redundant`: PR overlaps existing coverage without adding a clear new proof,
  contract, artifact, or durable decision.
- `readiness_treadmill`: PR mostly spent validation or queue effort without moving a research,
  evidence, workflow, or maintainability boundary enough to justify similar future routing.
- `blocked_or_inconclusive`: hindsight packet lacks enough evidence to classify progress without
  more source inspection or maintainer input.

### Relationship to originally-requested routing labels
Issue #3795 first named a routing-judgment set (`evidence_yielding`, `useful_readiness`,
`readiness_treadmill`, `duplicate_or_overlapping`, `too_low_priority`, `needs_successor`). The
maintainer's follow-up scope (full / partial / duplicate / bounded-remainder successor handling)
made a progress-disposition taxonomy the primary axis, so the verdicts above replace those labels
while preserving their judgments:

- `evidence_yielding` maps to `complete_progress` or `partial_existing_parent` (state the durable
  proof, contract, or artifact in `routing_value`).
- `useful_readiness` is a `complete_progress`/`partial_existing_parent` verdict whose value is
  readiness rather than new evidence; record that nature in `routing_value`/`routing_cost`.
- `readiness_treadmill` and `needs_successor` are kept verbatim.
- `duplicate_or_overlapping` maps to `duplicate_or_redundant`.
- `too_low_priority` is not a separate verdict; record a "correct work, but routing priority was too
  low" judgment in `routing_cost` alongside the chosen progress verdict.

## Successor Issue Packet
When the verdict is `needs_successor`, include a draft successor issue packet without posting it by
default. The packet must include:

- parent issue and PR reference;
- what was already completed;
- what remains;
- why the remainder is dissertation- or evidence-relevant;
- first bounded Codex implementation slice;
- validation expectation;
- why the slice is not duplicate completed coverage.

Queue and scout policy should exclude completed parent issues, but it may route an explicitly named
successor issue when the hindsight packet explains the remaining non-duplicate slice.

## Workflow
1. Confirm the caller's authorization and keep the review read-only unless mutation is explicitly
   authorized.
2. Fetch or inspect each PR packet: PR body, changed files, commits, linked issues, labels, merge
   state, validation evidence, and explicit exclusions.
3. Inspect linked issue state only as needed to decide full, partial, duplicate, or successor
   disposition.
4. Classify the PR using the verdict taxonomy and successor judgment.
5. For partial work, name the remaining bounded slice explicitly. If the parent issue remains open,
   prefer `partial_existing_parent` over a new successor recommendation.
6. Record caveats when the PR body or issue text is incomplete, compressed, or ambiguous.
7. Output a compact retrospective table plus any successor issue packets.

## Guardrails
- Do not treat merged status, passing checks, or `merge-ready` labels as proof that the linked issue
  was fully satisfied.
- Do not classify fallback or degraded benchmark execution as benchmark-strengthening evidence.
- Do not promote diagnostic-only results into paper, dissertation, benchmark, or claim-map changes.
- Do not create successor issues by default; recommend draft packets only.
- Do not close parent issues or edit queue metadata during hindsight review.
- Do not invent linked issue closure when the PR body says the work is support-only, partial, or
  leaves existing issues open.
- Keep confidence explicit when issue bodies, PR bodies, or validation logs are unavailable.

## Output
Return a compact `skill_run_summary.v1`-compatible summary with:

- reviewed PR list;
- per-PR verdict, successor judgment, bounded remainder, confidence, and evidence;
- successor issue packets for every `needs_successor` verdict;
- residual risks and missing evidence;
- confirmation that default read-only restrictions were preserved.
