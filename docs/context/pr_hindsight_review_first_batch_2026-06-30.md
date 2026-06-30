# PR Hindsight Review First Batch

Issue: [#3795](https://github.com/ll7/robot_sf_ll7/issues/3795)
Skill: [pr-hindsight-review](../../.agents/skills/pr-hindsight-review/SKILL.md)
Date: 2026-06-30

This note records the first read-only hindsight packet for four merged autonomous-routing pull requests (PRs): [#3775](https://github.com/ll7/robot_sf_ll7/pull/3775), [#3778](https://github.com/ll7/robot_sf_ll7/pull/3778), [#3779](https://github.com/ll7/robot_sf_ll7/pull/3779), and [#3791](https://github.com/ll7/robot_sf_ll7/pull/3791).

Evidence status: retrospective workflow evidence only. This note does not promote benchmark, dissertation, paper, or release claims.

## Cross-PR Pattern

This batch mostly produced useful readiness infrastructure: explicit metric semantics, compact frozen-trace status summaries, user-facing Social Navigation Quality Index (SNQI) governance alignment, and Package B registry preflight checks. The main treadmill risk is that three PRs improved readiness around still-open parent issues, so future routing should prefer the remaining decision or execution slices over another support-only polish pass.

No successor issue is recommended from this hindsight note because the bounded remainders are already tracked by open parent issues. Queue scouts should exclude closed parent issue #3724 unless a future review names a fresh non-duplicate successor slice.

## PR #3775

- `pr`: [#3775](https://github.com/ll7/robot_sf_ll7/pull/3775)
- `title`: `fix(benchmark): freeze pedestrian metric semantics`
- `state`: `merged`
- `merged_at`: 2026-06-28T18:28:13Z
- `linked_issues`: [#3724](https://github.com/ll7/robot_sf_ll7/issues/3724)
- `route_source`: ready-queue autonomous implementation for issue #3724.
- `scope_summary`: clarified collision and near-miss semantics across benchmark, SNQI proxy, validation, and tests.
- `validation_evidence`: PR validation and checks as recorded on PR #3775; no benchmark campaign evidence claimed here.
- `out_of_scope`: no benchmark campaign, graphics processing unit (GPU) work, Slurm work, release, or claim promotion.
- `issue_disposition`: PR appears to fully close issue #3724.
- `successor_judgment`: `no_successor_needed`
- `routing_value`: useful benchmark-safety contract work that reduces semantic ambiguity before future benchmark interpretation.
- `routing_cost`: touched shared metric and proxy paths and required focused validation.
- `routing_lesson`: keep routing high for semantic footguns that can mislead later benchmark interpretation, even when the PR itself is not benchmark evidence.
- `follow_up_action`: `no_action`
- `verdict`: `complete_progress`
- `routing_label`: `evidence_yielding`
- `confidence`: `medium`; this would change if downstream metric consumers still use an unaligned collision definition not covered by issue #3724.

## PR #3778

- `pr`: [#3778](https://github.com/ll7/robot_sf_ll7/pull/3778)
- `title`: `bench: summarize frozen trace artifact statuses`
- `state`: `merged`
- `merged_at`: 2026-06-28T18:32:05Z
- `linked_issues`: [#3482](https://github.com/ll7/robot_sf_ll7/issues/3482)
- `route_source`: autonomous support slice under issue #3482.
- `scope_summary`: added an aggregate affected-artifact summary to already-materialized frozen-trace reconciliation reports.
- `validation_evidence`: PR validation and checks as recorded on PR #3778; no frozen v0.0.2 reconciliation rerun claimed here.
- `out_of_scope`: did not locate artifacts, run old/new reconciliation, or decide claim, table, or figure validity.
- `issue_disposition`: useful partial coverage; issue #3482 remains open for the empirical-integrity blocker.
- `successor_judgment`: `existing_parent_issue_remains_open`
- `routing_value`: useful diagnostic reporting helper; reviewers can see affected claim, table, and figure status counts without manual aggregation.
- `routing_cost`: support-only readiness work.
- `routing_lesson`: route one support PR when it removes review friction, then route the empirical parent issue instead of repeating report-shape polish.
- `follow_up_action`: `keep_parent_issue_open`
- `verdict`: `partial_existing_parent`
- `routing_label`: `useful_readiness`
- `confidence`: `high`

## PR #3779

- `pr`: [#3779](https://github.com/ll7/robot_sf_ll7/pull/3779)
- `title`: `docs: align SNQI governance guide`
- `state`: `merged`
- `merged_at`: 2026-06-28T18:44:10Z
- `linked_issues`: [#3723](https://github.com/ll7/robot_sf_ll7/issues/3723), [#3699](https://github.com/ll7/robot_sf_ll7/issues/3699)
- `route_source`: autonomous documentation follow-up after SNQI diagnostics and preflight PRs.
- `scope_summary`: aligned the SNQI weight-tool guide with merged governance diagnostics and preflight work.
- `validation_evidence`: PR validation and checks as recorded on PR #3779; no canonical SNQI weight-source decision claimed here.
- `out_of_scope`: did not choose a canonical SNQI weight source or normalize mixed-basis SNQI terms.
- `issue_disposition`: useful partial coverage; issues #3723 and #3699 remain open for policy decisions.
- `successor_judgment`: `existing_parent_issue_remains_open`
- `routing_value`: useful readiness documentation because it kept user-facing guidance consistent with merged diagnostic and preflight behavior.
- `routing_cost`: docs-only alignment.
- `routing_lesson`: documentation alignment is worthwhile after governance behavior lands, but queue scouts should next prefer the explicit product decisions in issues #3723 and #3699.
- `follow_up_action`: `keep_parent_issue_open`
- `verdict`: `partial_existing_parent`
- `routing_label`: `useful_readiness`
- `confidence`: `high`

## PR #3791

- `pr`: [#3791](https://github.com/ll7/robot_sf_ll7/pull/3791)
- `title`: `Tighten Package B registry preflight issue #3079`
- `state`: `merged`
- `merged_at`: 2026-06-29T07:27:50Z
- `linked_issues`: [#3079](https://github.com/ll7/robot_sf_ll7/issues/3079)
- `route_source`: ready-queue autonomous support slice for issue #3079.
- `scope_summary`: tightened Package B registry preflight so it fails closed on missing required artifact registration and records registry provenance in metadata.
- `validation_evidence`: PR validation and checks as recorded on PR #3791; no budget-matched adversarial comparison or held-out interpretation claimed here.
- `out_of_scope`: did not execute the budget-matched adversarial comparison, certify candidates, count replayable failure yield, or interpret held-out-family results.
- `issue_disposition`: useful partial coverage; issue #3079 remains open for the actual Package B run and interpretation.
- `successor_judgment`: `existing_parent_issue_remains_open`
- `routing_value`: useful benchmark-readiness infrastructure; it makes a future Package B campaign harder to run against stale or malformed registry state.
- `routing_cost`: support/preflight changes with no benchmark campaign execution.
- `routing_lesson`: fail-closed preflight slices are valuable before compute-heavy benchmark work, but should hand back to the parent experiment once registry readiness is proven.
- `follow_up_action`: `keep_parent_issue_open`
- `verdict`: `partial_existing_parent`
- `routing_label`: `useful_readiness`
- `confidence`: `high`

## Successor Issue Judgment

No `needs_successor` packet is needed for this first batch. The only remaining bounded work is already tracked in open parent issues #3482, #3723, #3699, and #3079. If one of those parent issues closes without completing the named remainder, a future hindsight packet should include a draft successor issue with completed scope, remaining scope, dissertation or evidence relevance, first implementation slice, validation expectation, and non-duplication rationale.

## Routing Lessons

- Keep doing: route semantic-clarity and fail-closed preflight work when it prevents misleading benchmark interpretation or wasted compute.
- Stop doing: repeated support-only polish after the first review-friction reducer for a parent issue.
- Only do when: documentation alignment follows already-merged behavior and does not claim new benchmark, paper, or release evidence.
- New queue/scout rule: completed parent issues stay excluded unless a hindsight packet names a bounded, non-duplicate successor slice with explicit validation expectation.
