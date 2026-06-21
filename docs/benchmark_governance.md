# Benchmark Scenario And Model Governance

[Back to Documentation Index](./README.md)

This document defines the review contract for changing benchmark scenarios, metrics, model
profiles, and release-bound benchmark evidence. It complements the
[benchmark fallback policy](./context/issue_691_benchmark_fallback_policy.md), the
[artifact evidence vocabulary](./context/artifact_evidence_vocabulary.md), and the
[benchmark release protocol](./benchmark_release_protocol.md).

## Scope

Use this governance page for pull requests that add, remove, rename, promote, deprecate, or
semantically change any of the following:

- scenario suites, scenario contracts, scenario certificates, ODD declarations, seed policies,
  scenario matrices, maps used by benchmark suites, or benchmark scenario-selection rules;
- metric definitions, metric schemas, aggregate formulas, thresholds, normalization assets, report
  eligibility rules, leaderboard columns, or evidence-status classifications;
- model profiles, planner readiness profiles, learned-policy registry metadata, adapter contracts,
  observation tracks, model artifact pointers, or benchmark-promotion metadata;
- release-bound benchmark manifests, publication bundles, BenchmarkClaim artifacts, or
  paper-facing benchmark tables and figures.

Exploratory scripts, diagnostics, and local probes may move faster, but they must label themselves
as `exploratory`, `diagnostic-only`, `smoke evidence`, or `not benchmark evidence` when they do not
satisfy this contract.

## Review Rules

Every governed PR must make its claim boundary reviewable before interpretation. The PR description
or linked issue must state:

- the changed contract surface: scenario, metric, model profile, release evidence, or a combination;
- the intended evidence status: `diagnostic-only`, `smoke evidence`, `nominal benchmark evidence`,
  or `paper-grade`;
- whether the change is comparable with previous benchmark rows;
- fallback, degraded, adapter, excluded, or superseded rows that must not count as benchmark
  success;
- the exact validation command or artifact that proves the changed contract.

Reviewers should block merge-readiness when a PR silently changes benchmark semantics while only
presenting docs, generated figures, local `output/` files, or route evidence as proof.

## Scenario Governance

Scenario additions or modifications must include enough information for another contributor to
reproduce the intended benchmark boundary:

- stable scenario id, owning suite, and source path;
- map id or map source, start/goal route, actor population, randomization controls, horizon, and
  seed policy;
- intended use: development smoke, diagnostic stress case, nominal benchmark, release-bound
  benchmark, or paper-facing evidence;
- scenario-contract or certification status when the scenario can affect benchmark claims;
- changed assumptions relative to previous scenarios, including ODD, route-clearance, feasibility,
  pedestrian behavior, and termination semantics;
- expected impact on comparability, including whether old aggregate rows can still be compared.

Scenario PRs that promote a scenario into a benchmark suite must prove that the scenario is
loadable, eligible under the relevant certification rules, and executable through the intended
benchmark entry point. A scenario may be useful as a stress or diagnostic case without being
eligible for nominal or release-bound benchmark evidence.

## Metric Governance

Metric additions or modifications must define:

- metric id, schema version, units, denominator, valid range, missing-value behavior, and
  aggregation window;
- the episode fields consumed and whether the metric depends on planner outputs, simulator truth,
  derived traces, normalization assets, or external artifacts;
- threshold or weighting rationale, including the previous value when changing an existing metric;
- comparability impact for historical reports, leaderboards, and release-bound benchmark claims;
- fixture or sample proof that the metric computes the intended values and rejects malformed input.

Metric PRs must fail closed for invalid schemas, non-finite values, missing required provenance, and
mixed benchmark tracks unless the output is explicitly diagnostic. Fallback or degraded execution
can explain why a metric is unavailable, but it cannot be counted as successful benchmark evidence.

## Model Profile Governance

Model or planner-profile changes must keep the runtime contract and provenance explicit:

- stable model or profile id, profile schema version, planner family, adapter mode, and benchmark
  readiness status;
- observation schema or benchmark track, action schema, normalizer status, artifact URI, checksum,
  training config, training commit, and split contract when a learned artifact is involved;
- allowed benchmark suites and excluded suites, with a fail-closed reason for unavailable
  dependencies or artifacts;
- promotion boundary: `not_eligible`, `research_only`, `adapter_preflight`,
  `benchmark_candidate`, `benchmark_promoted`, or a stricter release-specific status;
- hydration and verification path for any model needed by public examples, release reproduction, or
  paper-facing claims.

Model profiles must not silently fall back to a different planner, checkpoint, normalizer, or
adapter and report that row as the requested model. Missing or mismatched artifacts must be
classified as `not_available`, `failed`, or another non-success status according to the benchmark
fallback policy.

## Versioning Rules

Use explicit schema/profile versions whenever a governed surface can affect reproducibility or
comparability. Version ids should be visible in the relevant YAML, JSON, registry entry, report, or
artifact manifest.

| Surface | Version marker | Patch change | Minor change | Major change |
| --- | --- | --- | --- | --- |
| Scenario schema | `scenario_schema_version` or the typed schema id, such as `scenario_contract.v1` | Documentation, typo, or metadata repair with no behavior change | Add optional fields, stricter provenance, or new diagnostics that preserve old rows | Change required fields, scenario semantics, seed policy, eligibility rules, or comparability |
| Metric schema | `metric_schema_version`, metric id suffix, or report schema id | Metadata repair or implementation bugfix that preserves intended values | Add optional metric fields, diagnostics, or stricter validation while preserving old metric meaning | Change formula, denominator, threshold, weighting, normalization basis, or aggregation semantics |
| Model profile | `model_profile_version`, `track_schema_version`, registry schema, or profile id suffix | Provenance repair, checksum repair, or docs repair | Add optional provenance, eligibility metadata, or verification fields | Change observation/action contract, adapter semantics, artifact identity, readiness status meaning, or benchmark eligibility |
| Release evidence | benchmark protocol version, release id, BenchmarkClaim schema, or manifest version | Provenance or packaging repair with identical benchmark contract | Comparable contract extension or stricter reproducibility metadata | Non-comparable scenario suite, planner set, seed policy, metric contract, model profile, or SNQI normalization change |

Do not reuse a version id for a semantic change. If a report or artifact combines multiple schema
versions, it must make the combination visible and classify cross-version comparisons as diagnostic
unless a compatibility note proves they are comparable.

## Release-Bound Evidence

Release-bound or paper-facing benchmark evidence requires:

- frozen benchmark contract: scenario suite, seed policy, planner/model profiles, metric schema,
  normalization assets, and benchmark protocol version;
- source commit, command/config path, dependency mode, environment assumptions, and artifact
  checksums;
- schema-checked episode records and aggregate reports stored as durable evidence, release
  artifacts, or tracked compact evidence copies, not only worktree-local `output/`;
- explicit exclusion of fallback, degraded, failed, not-available, and diagnostic-only rows from
  benchmark-success counts;
- reproduction instructions that a reviewer can run or audit, including hydration of external data,
  model artifacts, and normalization assets;
- evidence classification at the top of the report before rankings, recommendations, or success
  language.

If any release-bound input cannot be hydrated, validated, or reproduced, the evidence status is
`blocked`, `diagnostic-only`, or `not benchmark evidence` until the missing proof is supplied.

## Deprecated Or Superseded Scenarios

Do not delete benchmark-relevant scenarios silently. Deprecation or supersession must identify:

- the deprecated scenario id and source path;
- replacement scenario id, if one exists;
- reason: duplicate, infeasible, malformed, ODD mismatch, route-clearance failure, superseded by a
  better-controlled variant, or no longer aligned with the benchmark claim;
- effective version or PR where the change starts;
- historical report handling: keep comparable old rows, mark rows as superseded, exclude rows from
  new aggregates, or rerun the replacement matrix;
- whether old evidence remains valid for its original claim boundary.

Deprecated scenarios should remain discoverable through docs, suite metadata, registry notes, or a
context note until all active reports that cite them have either been updated or explicitly frozen
as historical evidence. Superseded scenarios may support historical interpretation, but they must
not be mixed into new release-bound aggregates unless the release manifest says so.

## Minimum PR Checklist

Before treating a governed PR as ready, verify:

- changed links, paths, schema/profile ids, and artifact pointers resolve;
- the PR labels the evidence tier and comparability impact;
- scenario, metric, and model-profile changes have explicit versioning decisions;
- release-bound evidence names durable artifacts and reproduction requirements;
- deprecated or superseded scenarios have documented replacement and historical-row handling;
- fallback/degraded rows are excluded from benchmark-success language;
- validation matches the risk tier in `docs/maintainer_values.md`.

Docs-only governance edits normally use the cheap validation path: inspect the diff and verify
changed links or referenced paths. Runtime, schema, metric, benchmark, model-provenance, or
paper-facing changes need executable proof appropriate to the claim.
