# Code Review Guide

Use this review guide for benchmark, planner, training, and documentation changes that can affect
published claims, reproducibility, or agent workflows.

## Review Priorities

Check these in order:

1. Correctness of user-visible behavior and public contracts.
2. Benchmark credibility and provenance.
3. Proof that the new or changed behavior actually works in this repository.
4. Reproducibility of commands, configs, and artifacts.
5. Scope discipline: no hidden workflow churn or speculative infrastructure.
6. Downstream propagation completeness for evidence-producing PRs: parent issue, claim map or
   benchmark report, leaderboard or artifact catalog, registry, context index, and follow-up issue
   updates are either done or explicitly deferred.
7. Test and documentation coverage proportional to the risk of the change.

Reject changes that only add code or docs without task-appropriate proof.

## Post-Merge Review-Thread Sweep

Merge does not mean "all review findings adjudicated." Inline review threads left unresolved
on a merged PR are easily lost, and some are later fixed by successor PRs without ever being
linked. After merge, run a short sweep so the thread ledger reflects reality:

1. Re-query unresolved review threads on the merged PR (`gh api graphql` on `reviewThreads`,
   filtering `isResolved == false`).
2. For each unresolved thread, either resolve it (if the merge or a successor commit addressed
   the comment) or link the fixing PR/commit/issue in a one-line reply. Threads whose substance
   is still open get a dedicated follow-up issue instead of a silent resolve.
3. Record the sweep outcome in the merge note or the issue thread so the count of unresolved
   threads is auditable.

Treat substantive threads that need real work (for example a fail-closed contract gap or an
end-to-end assertion gap) as follow-up issues, not as items to resolve with a link. This sweep is
bookkeeping: it does not change benchmark, metric, or schema semantics.

## Intended Design Alignment

Before treating CI or targeted tests as sufficient, compare the PR against the linked issue,
design note, PR body, changed behavior, tests, docs, and stated claims.

Reviewers should explicitly classify the result:

- `aligned`: implementation behavior, tests, docs, and claims satisfy the intended design or issue
  contract.
- `intentionally narrowed`: the PR solves a useful subset, names the narrowed scope, and links
  follow-up issues for real deferred work.
- `blocked`: required contract, public-claim, benchmark, schema, metric, or runtime-safety work is
  missing and must be fixed before merge.
- `handoff-only`: remaining information is transient state, CI waiting, cleanup context, or local
  reviewer notes with no durable action.

Create a follow-up issue only for deferred work that is actionable outside the current PR. A good
follow-up issue names:

- the residual risk or deferred behavior,
- why it should not block the current PR,
- the acceptance condition or stop rule,
- the expected validation or proof tier,
- links back to the PR, issue contract, design note, or evidence that revealed it.

Do not use generic backlog notes as substitutes for blockers or for follow-up issues with a clear
acceptance condition.

## Domain-Aware Approval Gate

Passing CI, automated review, and implementation-integrity checks is not sufficient for PRs that
change evidence classification, experimental comparison methodology, figure eligibility, benchmark
interpretation, or paper-facing claim surfaces. Those PRs need an explicit `Domain-Aware Approval`
section in the PR body before they can be treated as merge-ready.

The gate is intentionally narrow. Ordinary implementation, docs, and test-only PRs can mark the
domain approval requirement as not applicable when they do not alter evidence validity, comparison
methodology, figure eligibility, or claim boundaries.

For PRs where the gate applies, reviewers should verify that the PR body names:

- target claim or hypothesis;
- comparator and split/evidence-validity policy;
- fallback/degraded exclusions;
- claim boundary and evidence tier;
- whether the change proves implementation integrity only, or also supports experimental validity.

PR #3276 is the motivating comparison-methodology example: automated review accepted the linked
issue framing even though the comparison remained circular and lacked the held-out scenario-family
design required by the issue. PR #3273 is the motivating evidence-classification example:
machine-readable classifications and conservative prose diverged. Future PRs in those categories
should be held until a domain-aware approval note is present, or should state a blocker instead of
being presented as merge-ready.

## Benchmark-Credibility Review

For benchmark-facing changes, explicitly verify:

### Evaluation semantics
- Does the change alter success, collision, timeout, or metric semantics?
- Are fallback, skip, and fail-fast paths still explicit and testable?
- Are evidence statuses (`diagnostic-only`, `smoke evidence`, `nominal benchmark evidence`,
  `paper-grade`) still accurate?
- Does any report wording overstate what the benchmark actually measures?
- For mixed, partial, or fallback-tainted evidence, does the report start with claim boundary,
  evidence status, fallback/degraded exclusions, major caveats, and uncertainty before ranking or
  success language?

### Observation normalization
- Are observation keys, bounds, clipping, and dtype contracts preserved or versioned?
- If a learned policy is involved, does the runtime observation contract still match the training contract?
- Are adapter transformations documented when planner inputs differ from env-native observations?

### Scenario distributions
- Does the scenario set, seed policy, or map pool change what is being compared?
- Are scenario family changes reflected in configs, docs, and interpretation guidance?
- Does any new default silently bias benchmark outputs toward a subset of scenarios?

### Reproducibility
- Is there a committed config or canonical command for the new behavior?
- Are generated outputs still rooted under `output/`?
- Do docs and PR text identify the exact command path needed to reproduce the result?
- If results depend on optional extras, hardware, or third-party assets, is that dependency explicit?

### Upstream provenance
- For vendored, wrapped, or adapter-backed planners: is the upstream source still identifiable?
- Are licenses, checkpoints, model origins, and wrapper boundaries documented?
- Does the change preserve the claim that a result is original-code-backed rather than a local reimplementation?

## Planner Integration Review

For planner additions or modifications, review:

- input contract: required observation/state fields are explicit,
- output contract: action space and kinematics adaptation are explicit,
- fallback policy: missing dependencies or artifacts fail clearly,
- provenance: upstream repo/model references remain auditable,
- proof: benchmark or targeted runtime evidence shows the planner actually runs in this repository,
- benchmark readiness: paper-facing vs experimental status is still correctly labeled.

If a planner is added but never executed in the local benchmark stack, treat the change as
incomplete.

## Proof Requirements By Change Type

- new planner or planner integration:
  run a benchmark, policy-analysis path, or equivalent executable check that proves integration
  works here.
- metric update:
  provide targeted assertions, fixtures, or sample outputs that prove the new metric behavior.
- new skill:
  verify the referenced files, commands, and discoverability path are correct for this repository.
- new test:
  show that it protects a real contract or regression and is not only syntactic coverage.
- docs-only guidance:
  verify every referenced path and command, and ensure the guidance matches the current repo
  workflow.
- new research, benchmark, metric, or paper-facing analysis tool:
  require one representative use on durable/versioned input, or a linked follow-up issue that names
  the decision, claim boundary, benchmark report, registry, context note, or synthesis surface the
  tool will update. Local-only `output/` files are not durable proof unless promoted or represented
  by a tracked manifest, registry entry, context note, or external artifact pointer. Small support
  helpers with no research-interpretation role may be marked `NA` with that reason.

## Evidence Artifact Fallback Review

Use this checklist for evidence-producing PRs when automated AI review is rate-limited, unavailable,
or path-filtered in a way that may skip small durable evidence files such as CSV tables under
`docs/context/evidence/`.

- Confirm every linked issue, PR, context note, config, script, and evidence path resolves from a
  fresh checkout.
- Confirm copied evidence files are intentionally small and reviewable. Large raw logs, videos,
  model caches, coverage HTML, and raw episode JSONL should stay out of git unless a narrow fixture
  reason is stated.
- For CSV evidence, inspect the header and at least one representative row. Verify column names,
  units, seed/scenario/planner identifiers, and status fields match the surrounding Markdown or
  JSON summary.
- Check that parent issue conclusions, PR text, and context-note wording agree on result
  classification: benchmark evidence, smoke evidence, diagnostic-only, blocked, or proposal.
- Check claim-boundary language explicitly separates observed evidence from hypotheses, future
  work, fallback/degraded execution, and paper-facing claims.
- Check fallback/degraded rows are introduced as caveats or exclusions before any comparative
  ranking, aggregate success language, or recommendation.
- Check generated evidence points back to a reproducible command, commit, config or scenario
  matrix, seed policy, and durable artifact/provenance decision.
- If an evidence CSV or other small table is path-filtered out of automated review, state in the PR
  review or merge note that this manual fallback checklist covered it.

## Documentation Review

Docs changes should be rejected if they:

- introduce benchmark claims without a reproducible command path,
- blur the line between observed evidence and future intent,
- describe experimental planners as baseline-ready,
- omit caveats around optional dependencies or upstream source limits.
- bury mixed-evidence claim boundaries or sub-95% confidence caveats after result interpretation.

Prefer docs that point to canonical config files, scripts, and issue execution notes instead of freehand operational prose.

## Tests And Validation

Apply Principle XIII from `.specify/memory/constitution.md`:

- fix high-signal tests immediately,
- challenge flaky or low-value tests before investing in them,
- add tests when public contracts or benchmark semantics change,
- require proof that new tests or fixes meaningfully validate the intended contract,
- keep validation commands in PR text explicit and reproducible.

For shared-helper consolidations, require a per-call-site contract table and tests for every
applicable behavior: return schema, missing and malformed input error/exit behavior, standalone
import footprint, read strategy, `path:line` context, and output ordering. For serialization,
subprocess, GPU-isolation, artifact-promotion, and CLI-handoff changes, require at least one
one-real-path test that invokes production's serializer and dispatch path without manually
pre-transforming the fixture.

See `docs/context/issue_1436_reproducibility_flaky_acceptance.md` for the canonical
classification of deterministic, environment-class, and stochastic failures, and the
explicit rerun boundary.

Recommended validation gate for broad changes:

```bash
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

For docs-only or context-stack changes, still verify the modified markdown paths and any repo-local skill references.
