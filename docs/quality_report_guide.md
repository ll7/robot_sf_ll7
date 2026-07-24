# Quality Report Guide

**Purpose**: Define the reproducible, commit- and environment-stamped quality
report that aggregates the repository's existing quality signals into one JSON
artifact. The report is for [issue #6213](https://github.com/ll7/robot_sf_ll7/issues/6213),
a child of the [quality-engineering epic #6205](https://github.com/ll7/robot_sf_ll7/issues/6205).

## Plain-language summary

The quality report collects signals that already exist in the repository — test
results, coverage, mutation testing, test duration, flakiness, contract and
scenario coverage, reproducibility, performance regression, and escaped defects
— and writes them into one `quality_report.v1` JSON file stamped with the exact
Git commit, branch, and environment that produced it. Its central design rule is
that it **never collapses distinct signals into a single "quality score"** and it
**never invents a metric or silently changes a gate**: each signal keeps its own
availability, source, role, and decision use, and every signal that cannot be
computed from an existing surface is recorded as `unavailable` or `deferred`
with a `source_gap` that names the missing surface.

## Why this exists

The repository already has a large pytest suite, a four-shard CI, coverage
tooling, a bounded mutation-testing lane, reproducibility checks, and
performance smokes (see [epic #6205](https://github.com/ll7/robot_sf_ll7/issues/6205)).
The gap is that these surfaces are not one discoverable, ownerable, reproducible
quality picture. This report closes that gap for the *aggregation* slice without
rewriting any test or CI lane, raising any threshold, or changing required-check
behavior.

## Hard design rules

These rules are enforced by the schema and the generator, and match the issue's
stop conditions:

1. **No vanity score.** There is no top-level single-number quality score.
   Diagnostic and gate signals are kept separate, each with a `role` of `gate`
   or `diagnostic`.
2. **No invented metrics; no silent gate changes.** When a signal cannot be
   computed from an existing authoritative surface, it is `unavailable` or
   `deferred` with a `source_gap` (reason + source + optional follow-up).
   Coverage thresholds, CI gates, and required checks are *read*, never
   *changed*.
3. **Separate categories stay separate.** Mutation `no_test`, `equivalent`,
   `timeout`, `survived`, and benchmark `fallback`/`degraded` counts are kept
   individually visible and are never folded into another category or counted as
   success.
4. **Reproducible.** The report records the exact producing command, the source
   surfaces read, full commit provenance, and a `tree_clean` flag.

## The report artifact

- **Schema**: [`robot_sf/benchmark/schemas/quality_report.schema.v1.json`](../robot_sf/benchmark/schemas/quality_report.schema.v1.json)
  (JSON Schema draft 2020-12).
- **Generator**: [`scripts/quality/generate_quality_report.py`](../scripts/quality/generate_quality_report.py).
- **Default output**: `output/quality/quality_report.json` (gitignored,
  worktree-local, regenerated per run — see
  [AGENTS.md "Worktree Teardown And Artifacts"](../AGENTS.md)).

### Quick start

```bash
# Default: read available surfaces, mark the rest unavailable/deferred.
uv run python scripts/quality/generate_quality_report.py

# With a coverage export, mutation baseline, and pytest JSON report:
uv run python scripts/quality/generate_quality_report.py \
  --coverage-json output/coverage/coverage.json \
  --mutation-baseline scripts/validation/mutation_baseline.json \
  --pytest-report output/test/pytest-report.json \
  --strict-reproducible true \
  --output output/quality/quality_report.json
```

The generator validates its own output against the schema and exits non-zero if
the report is invalid, so a written report is always a valid report.

## Schema structure

The top level records provenance, scope, cadence, inputs, and a `signals` block.
There is deliberately **no** score field at the top level.

| Field | Purpose |
| --- | --- |
| `schema` / `schema_version` | Discriminator (`quality_report.v1`) and semver. |
| `report_id` / `generated_at` | Stable instance id (commit + timestamp) and UTC time. |
| `scope` | `package`, `view` (`full_suite` / `changed_files` / `module_subset`), optional `baseline_ref` and `changed_files`. |
| `cadence` | `when` (`per_pr` / `scheduled` / `manual` / `release`), `owner`, `lane_command`. |
| `provenance` | `git_commit` (40 chars), `git_branch`, `git_dirty`, `python_version`, `key_packages`, `hardware`, `timestamp`. |
| `tree_clean` | `true` only when produced from a clean tree (`git_dirty == false`). A `false` value is a reproducibility caveat, not a failure. |
| `inputs` | `command` (exact reproducible command), `sources` (authoritative surfaces read), `seed`. |
| `signals` | The nine signal families described below. |
| `notes` | Optional free-form caveats; never a substitute for structured signals. |

## Signal families, formulas, and decision use

Every signal family has a `role` (`gate` or `diagnostic`), an `availability`
(`available` / `unavailable` / `deferred`), a `source` or `source_gap`, and a
`decision_use`. `available` requires a `source`; `unavailable`/`deferred`
requires a `source_gap`.

### 1. Test results (`signals.test_results`) — gate

- **pass_rate** = `passed / evaluated`, where
  `evaluated = passed + failed + errored` (skips excluded from the denominator).
- **collection_completeness** = `collected / selected`, where `collected`
  includes collected skipped/xfail tests and `selected` is the configured test
  selection; below 100% means a collection error or deselection.
- **counts**: `passed`, `failed`, `errored`, `skipped`, `xfailed`, `xpassed`,
  `warnings`.
- **Decision use**: a failed or errored test, or a collection error, blocks
  merge on the strict deterministic lane; skips do not.
- **Source**: a structured pytest report (e.g. `pytest --json-report`).
  Currently `unavailable` by default because the strict lane prints to the
  console but emits no structured summary.

### 2. Coverage (`signals.coverage`) — gate (line floor) / diagnostic (branch, changed-file)

- **line** = `covered_lines / num_statements * 100`.
- **branch** = `covered_branches / num_branches * 100` (diagnostic).
- **changed_files** = aggregate line coverage over the changed-file set vs the
  base ref, with `min_threshold` (80, from `scripts/dev/check_changed_coverage.sh`)
  and `goal_threshold` (100) surfaced as observed values.
- **Decision use**: the 85% absolute line-coverage floor is an enforced gate on
  non-PR CI (see [Coverage Guide](coverage_guide.md)). Branch and changed-file
  coverage are diagnostics unless a lane declares them. **No threshold is
  changed by this report.**
- **Source**: `coverage.py` JSON export (`--cov-report=json`), e.g.
  `output/coverage/coverage.json`.

### 3. Mutation (`signals.mutation`) — diagnostic (never per-PR gate)

- **categories** (kept separately visible):
  `killed`, `survived`, `equivalent`, `timeout`, `no_test`, `suspicious`,
  `skipped`, and `total_mutants`.
  - `equivalent`: surviving mutants reviewed and judged semantically equivalent
    — reported separately from `survived` so un-reviewed survivors are never
    hidden.
  - `timeout`: kept separate because timeouts may mask undetected faults.
  - `no_test`: generated mutants not exercised by any test — kept separate to
    expose coverage-of-assertions gaps.
- **mutation_score** = `killed / (total_mutants - equivalent) * 100`
  (`equivalent_excluded: true`), because equivalent mutants are not detectable
  faults. The score is omitted when the authoritative baseline does not provide
  an equivalent-mutant count; an omitted category is not inferred as zero.
- **baselined_survivors**: the downward-ratchet view when the baseline provides
  a survivor count. `new_unbaselined_survivors` requires a current mutmut
  comparison and is not emitted from a baseline alone. A non-zero value is a
  ratchet regression for the diagnostic lane, not a merge blocker.
- **Decision use**: mutation testing is a scheduled diagnostic only and is
  **never required per-PR before a baseline exists** (issue stop condition). See
  [mutation-testing triage](../mutation_testing_triage.md).
- **Source**: `scripts/validation/mutation_baseline.json` and
  `scripts/dev/mutation_ratchet.py`.

### 4. Test duration (`signals.test_duration`) — diagnostic (budget can be a gate)

- **total_seconds** and the **slowest_tests** list.
- **timeout_budget_compliance**: `over_budget` / `measured` against an optional
  per-test `budget_seconds`; `compliant` is true only when `over_budget == 0`.
- **Decision use**: surfaces slow tests and budget violations. Per-test budget
  compliance can become a gate when a lane declares a budget; by default it is
  advisory.
- **Source**: a structured pytest report with per-test durations.

### 5. Flakiness (`signals.flakiness`) — diagnostic

- **flaky_rate** = tests that failed at least once but passed on rerun /
  tests_run, over a rerun-enabled cycle.
- **rerun_rate** = `rerun_invocations / total_invocations`.
- **skip_xfail_age**: age of skip/xfail markers since their introducing commit,
  with a `stale_threshold_days` to flag stale suppressions for re-triage.
- **Decision use**: informs quarantine and re-triage; not a merge gate.
- **Status**: currently `unavailable` — no authoritative rerun/flaky summary or
  skip/xfail-age report is produced by an existing surface. Reruns are only
  permitted under the narrow policy in
  [issue #1436](context/issue_1436_reproducibility_flaky_acceptance.md).

### 6. Contract / scenario / hazard-ODD / compatibility coverage (`signals.contract_scenario_coverage`) — diagnostic

- **dimensions**: `contract_schema`, `scenario`, `hazard_odd`, `compatibility`,
  each `{covered, total, value, mapping_source}`.
- **Decision use**: maps protected behavior contracts to their protecting test
  level. Diagnostic until a traceability check produces machine-readable counts.
- **Status**: currently `unavailable` — a machine-readable contract-to-test
  traceability report does not yet exist. Scenario, ODD/hazard, and
  compatibility schemas exist under `robot_sf/benchmark/schemas/`, but their
  coverage by tests is not yet counted by an authoritative surface.

### 7. Reproducibility (`signals.reproducibility`) — gate (strict lane) / diagnostic (benchmark)

- **strict_lane_reproducible**: true when the deterministic strict lane
  reproduces for the same commit and environment (null when not measured).
- **benchmark**: `reproducible`, `fallback_count`, `degraded_count`,
  `native_count`, `total_runs`, `seed`. **Fallback and degraded counts are kept
  separate and are never counted as success evidence** (issue stop condition).
- **Decision use**: strict-lane reproducibility is a gate for deterministic
  contracts; benchmark reproducibility and fallback/degraded counts are
  diagnostics.
- **Source**: an explicit result from `scripts/benchmark_repro_check.py` when
  supplied. Without a result, the signal is `unavailable`; the generator does
  not report zero fallback/degraded/native runs from a missing surface.

### 8. Performance regression (`signals.performance_regression`) — diagnostic (advisory on PR, enforced on main)

- **regressions**: per-entry `{name, baseline_seconds, current_seconds,
  relative_change, enforced}`. `enforced` preserves whether the regression fails
  the lane in the current cadence.
- **regression_count**.
- **Decision use**: cold/warm regression is advisory on PRs and enforced on
  `main`/`workflow_dispatch`; the full performance smoke runs only in strict
  mode.
- **Source**: `scripts/validation/performance_smoke_test.py` (currently
  `unavailable` until a structured summary is wired in).

### 9. Escaped defects (`signals.escaped_defects`) — diagnostic

- **escaped_count** and **window** (e.g. "since last release"), plus
  **linked_issues**.
- **Decision use**: a diagnostic leading indicator for test-strategy gaps;
  recorded individually and never aggregated into a vanity score.
- **Status**: currently `unavailable` — no authoritative escaped-defect feed is
  wired in.

## Scope and cadence

- **Scope**: defaults to the full `robot_sf` suite. Use `--scope-view
  changed_files --baseline-ref origin/main` for a per-PR changed-file view.
- **Cadence**: the report is produced on demand (`--cadence manual`) by default.
  A per-PR run is **advisory only**; mutation testing and long benchmark
  campaigns stay `scheduled`/`manual` and are **never** forced per-PR before a
  baseline exists (issue stop condition). A `release` cadence aggregates the
  strongest evidence for release readiness.
- **Owner**: `quality-engineering`.

## Reproducibility and the clean-tree check

The report records `provenance.git_commit` (40-character hash),
`provenance.git_branch`, `provenance.git_dirty`, `provenance.python_version`,
`provenance.key_packages`, and a `hardware` profile. `tree_clean` is the
negation of `git_dirty`. A report from a dirty tree is still valid but carries a
reproducibility caveat; for release-grade evidence, produce the report from a
clean tree at a known commit.

The `inputs.command` field records the exact command that produced the report,
and `inputs.sources` lists the authoritative surfaces read, so the report can be
regenerated deterministically from the declared inputs.

## Proportionality: diagnostic vs gate

| Signal | Default role | Blocks merge? |
| --- | --- | --- |
| Test pass rate, collection completeness | gate | yes (strict lane) |
| Line coverage (85% floor) | gate | yes (non-PR CI) |
| Branch coverage | diagnostic | no |
| Changed-file coverage | diagnostic | no (unless a lane enforces) |
| Mutation | diagnostic | no (scheduled lane only) |
| Test duration / budget | diagnostic | no (unless a budget is declared) |
| Flakiness, skip/xfail age | diagnostic | no |
| Contract/scenario/hazard-ODD/compatibility | diagnostic | no |
| Benchmark reproducibility, fallback/degraded | diagnostic | no |
| Performance regression | diagnostic (advisory on PR) | enforced on main/dispatch only |
| Escaped defects | diagnostic | no |

This table is the contract for not silently promoting a diagnostic into a gate.

## What is explicitly out of scope

- Rewriting the test or CI workflow.
- Raising coverage thresholds or changing required-check behavior.
- Making mutation testing or long benchmark campaigns required on every PR.
- Creating a single vanity quality score.
- Changing simulator, planner, benchmark, metric, schema, or publication
  semantics.

## Related documents

- [Coverage Guide](coverage_guide.md) — coverage collection, baseline tracking,
  absolute floor enforcement, and CI integration.
- [mutation-testing triage](../mutation_testing_triage.md) — mutation lane
  status, survivor triage, and the downward ratchet.
- [Issue #1436 reproducibility/flaky acceptance policy](context/issue_1436_reproducibility_flaky_acceptance.md)
  — deterministic lanes and the narrow rerun policy.
- [Development Guide](dev_guide.md) — setup, workflow, testing, and CI.
