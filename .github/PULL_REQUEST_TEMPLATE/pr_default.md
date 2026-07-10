## Summary
What changed, in one or two sentences.

## Linked Issues
- Closes `#<id>`
- Relates to `#<id>`

## Stack / Dependency
- Base dependency: none / `PR #<id>` / branch `<name>`
- Required prior PRs: none / `#<id>`
- Stack follow-up issues: none / `#<id>`
- Safe to review independently: yes / no
- Review dependency reason, if any: `<reason>`

## What Changed
- Key implementation changes.
- Any docs, config, or tests updated.

## Why It Matters
- Added value:
- Expected impact:
- Why this is worth merging now:

## Research Result Guidance
Required for research-labelled, benchmark-labelled, metric-facing, paper-facing, or other
evidence-producing PRs. For support/tooling/docs-only PRs that make no research claim, set fields
to `NA` and state why.

- Target claim / hypothesis / blocker this should affect:
- Comparator or baseline, if applicable:
- Evidence tier: full benchmark / targeted smoke / diagnostic probe / launch packet / docs-only / NA
- Result classification: positive / negative / inconclusive / diagnostic-only / blocker-resolution / NA
- Decision or stop rule, if applicable:
- Parent issue, claim map, registry, context note, or synthesis surface to update:
- New research/benchmark/metric/paper-facing analysis tool, if any: representative use on
  durable/versioned input, or a linked follow-up issue that names the decision, claim boundary, or
  synthesis surface it will update. Examples: trace-panel generators, topology-score
  instrumentation, seed-sufficiency analysis, and why-report generation. For support helpers that
  do not interpret research evidence, state `NA - support helper` with the reason:

## Domain-Aware Approval
Required when `Evidence tier` is not `NA`/`docs-only`, or `Result classification` is not `NA`.
Use this for PRs that change evidence classification, experimental comparison methodology, figure
eligibility, benchmark interpretation, or paper-facing claim surfaces. This approval concerns
experimental validity and claim boundaries; normal implementation integrity still comes from code
review, tests, and CI. Docs-only/support PRs that only *mention* an evidence concept in prose (and do
not fill a concrete `Evidence tier`/`Result classification`) may opt out with
`Required for this PR: no - reason` and `Status: not required`.

- Required for this PR: yes / no - reason
- Domains reviewed: evidence classification / experimental comparison / figure eligibility / benchmark interpretation / paper-facing claims / NA
- Status: approved / waived / pending / blocked / not required
- Approver/review source or waiver:
- Validity checklist:
  - Target claim/hypothesis:
  - Comparator or split/evidence validity:
  - Fallback/degraded exclusions:
  - Claim boundary:
  - Implementation integrity vs experimental validity:

## Falsification / Non-Transfer Check
Required for research-result PRs when the expected mechanism did not improve the measured outcome,
only helped a local slice, or produced diagnostic-only evidence. For support/tooling/docs-only PRs
that make no research claim, set fields to `NA` and state why.

- Did the mechanism activate? yes / no / unknown / NA
- Did the intervention change command source, selected command, trajectory, or route progress?
- Did the scenario actually contain the targeted failure mode?
- Result route: stop / revise / narrow / continue / NA
- Follow-up question or issue for weak, negative, or non-transfer results:

## Next Empirical Action
Required when evidence is missing, blocked, diagnostic-only, or negative. For docs-only, support,
or already-complete evidence PRs, set fields to `NA` and state why.

- Rerun needed: yes / no / NA
- Extractor or analysis tool needed: yes / no / NA
- Artifact missing or unavailable:
- Stop / revise / continue decision:
- Proposed child issue or existing follow-up:

## Validation / Proof
For research/benchmark/metric/paper-facing analysis-tool PRs: include one
representative use on durable/versioned input (tracked config, model
checkpoint, committed fixture, or versioned W&B artifact), or link a
concrete follow-up issue that names the decision, claim boundary, or
synthesis surface the tool will update. Local-only `output/` files are
not durable proof unless promoted or represented by a tracked manifest.
Small support helpers (formatters, CLI wrappers, quick diagnostics)
that make no research/benchmark/metric/paper claim are exempt.
Examples that do need first use or a concrete follow-up include trace-panel generators,
topology-score instrumentation, seed-sufficiency analysis, and why-report generation. Support
helpers that do not interpret research evidence should state `NA - support helper` with the reason.

- Commands run:
- Evidence that the change works here:
- Benchmarks or smoke tests, if applicable:

## Performance Evidence
Required for `perf`-typed changes (conventional-commit `perf(...)`); delete this section
otherwise. Enforced by `scripts/dev/check_perf_evidence.py` so a claimed speed-up is
substantiated on the real entry point (see the #3611 → #3613 wrong-layer revert). Use concrete
values, not `NA`. The cache field is required only when the change claims caching/reuse.

- Baseline runtime: `<time>` on `<representative slice + seeds>`
- Changed runtime: `<time>` on the same slice + seeds
- Representative command: `<command that reproduces the measurement>`
- Hot-path call count or profile anchor: `<profile/counter on the real campaign entry point>`
- Cache-hit or reuse counter: `<hits/misses or reuse count, or NA when no caching is claimed>`
- Rollback or failure criterion: `<observable regression that triggers a revert>`

## Risks / Rollout
- Compatibility risks:
- Failure modes:
- Rollback or fallback plan:

## Docs / Provenance
- Updated docs:
- Relevant design or provenance notes:
- Any assumptions that need to be preserved:

## Downstream Propagation
- Parent issue updated (yes/no/NA):
- Claim map / benchmark report updated (yes/no/NA):
- Leaderboard / artifact catalog updated (yes/no/NA):
- Registry or config index updated (yes/no/NA):
- Context index / memory note updated (yes/no/NA):
- Follow-up issue opened for deferred propagation (yes/no/NA):
- Not applicable because:

For benchmark, registry, claim-map, or paper-facing changes, explicitly say when downstream
propagation is deferred and name the issue or note that will carry it.

## Follow-Up Issues
- Deferred work:
- Issues opened for follow-up:

## Reviewer Notes
- Anything a reviewer should verify closely:
- Any known limitations:
- Shared-helper migration: include a per-call-site contract table (return schema, missing/malformed
  behavior and exit code, import footprint, read strategy, `path:line` context, ordering), or state
  why a row is inapplicable. Process-boundary changes need one test on the real production path.
