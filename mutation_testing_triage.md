# Mutation-testing triage (issue #5508)

> **Status: scheduled diagnostic, never a required PR gate, not benchmark
> evidence.** This document records the triage of surviving mutants produced by
> the bounded mutation-testing lane over `robot_sf/research/aggregation.py`
> exercised by `tests/research/test_aggregation.py`. It supersedes the
> first-slice triage from PR #5513 and reflects the scheduled-automation +
> downward-ratchet slice.

## Why this exists

CI enforces an 85% line-coverage floor, but line coverage does not establish
that assertions **detect** incorrect metric, provenance, or publication-gate
behavior. Mutation testing measures exactly that: it mutates the source and
checks whether the test suite catches the change. This lane is bounded to one
benchmark-critical module (`aggregation.py`) so the cost stays predictable, and
it runs weekly + on manual dispatch instead of on every pull request.

## Reproduction

From the repository root, with the development dependencies installed:

```bash
# Re-run mutmut over the configured module and gate against the baseline.
uv run python scripts/dev/mutation_ratchet.py --check

# Or run mutmut directly (the [mutmut] config in setup.cfg pins the scope):
uv run mutmut run
uv run mutmut results
```

The `[mutmut]` configuration in `setup.cfg` pins:

- source: `robot_sf/research/aggregation.py`;
- copied runtime package: `robot_sf`;
- selected tests: `tests/research/test_aggregation.py`;
- `debug=true` so mutmut emits the `mutants/mutmut-cicd-stats.json` summary.

## The downward ratchet

The committed baseline of tolerated survivors lives at
`scripts/validation/mutation_baseline.json`. The ratchet
(`scripts/dev/mutation_ratchet.py --check`, run by
`.github/workflows/mutation-testing.yml`) fails **only** when a **new,
un-baselined** surviving mutant appears. Survivors already listed in the
baseline are tolerated grandfathered exclusions; survivors that disappear
(killed by new assertions) are reported as **ratchet opportunities** and can be
locked in by refreshing the baseline:

```bash
# After adding assertions that kill a baselined survivor:
uv run python scripts/dev/mutation_ratchet.py --write-baseline
```

The baseline only ever shrinks: a killed survivor is dropped on refresh.
Surviving mutants that are flaky or environment-dependent (`timeout`,
`suspicious`, `skipped` states) are intentionally excluded from the baseline;
only stable `survived` mutants are tracked.

## Current result

| Classification | Count | Interpretation |
| --- | ---: | --- |
| Killed | 391 | Selected tests detected the mutation. |
| Survived | 32 | Enumerated in the baseline; triaged below. |
| No tests | 0 | All generated mutants are now exercised. |
| Timeout / suspicious / skipped | 0 | No execution anomaly was observed. |
| Total | 423 | All generated mutants in this run. |

PR #5513 (first slice) reported 200 killed / 5 survived / 218 no-test out of
423 mutants. The second slice expanded `tests/research/test_aggregation.py`
with targeted tests for the export, manifest, and extraction paths, moving the
218 no-test mutants into the executable set. The current run therefore shows
32 survivors across 7 functions; all are baselined and tolerated.

The third slice (issue #6122) strengthened assertions to kill 23 more survivors:
the `metrics not found` reason was tightened from a substring to an exact-
equality check, and per-alias / per-optional-field tests were added for the
`extract_seed_metrics` fallback chain (`avg_timesteps`, `total_timesteps`,
`final_reward_mean`, `run_duration_seconds`). The remaining 32 survivors are
the genuinely equivalent mutants triaged below (encoding platform-default
equivalence, logging-only kwargs, and `None`-for-default fallbacks).

## How to triage a survivor

Inspect any baselined survivor with:

```bash
uv run mutmut show <survivor-name>
# e.g.
uv run mutmut show robot_sf.research.aggregation.x_extract_seed_metrics__mutmut_32
```

Classify it as **(a) a missing assertion worth adding** or
**(b) an equivalent mutant to keep baselined**, using the categories below.

## Survivor categories

### Equivalent mutants (from first-slice review, still baselined)

These 5 survivors were reviewed as equivalent under the current contract in
PR #5513 and remain baselined:

| Mutants | Disposition |
| --- | --- |
| `aggregate_metrics` 18 and 19 | Equivalent for the current schema: `policy_type` is the grouping metadata field and supported values are non-numeric. Revisit if numeric `policy_type` values become supported. |
| `aggregate_metrics` 59; `bootstrap_ci` 13 | Equivalent: omitting `replace=True` uses NumPy's default replacement behavior. |
| `compute_completeness_score` 3 | Equivalent: changing the non-numeric sort priority from `1` to `2` does not change ordering because numeric keys use `0` and non-numeric keys share the same priority. |

### `None` substituted for a keyword default (equivalent mutant)

The most common class in the newly-exercised paths. mutmut replaces a keyword
argument's value with `None`:

| Survivor (example) | Mutation | Why it survives |
| --- | --- | --- |
| `x_export_metrics_json__mutmut_10` | `encoding="utf-8"` -> `encoding=None` | `open(..., encoding=None)` uses the platform default, which is UTF-8 on the CI runner and on most dev machines; the JSON round-trip test does not distinguish. |
| `x__load_manifest_payload__mutmut_2` | `encoding="utf-8"` -> `encoding=None` | Same platform-default equivalence for `read_text`. |
| `x_export_metrics_csv__mutmut_1` | `parents=True` -> `parents=None` | `parents=None` is falsy; the affected paths resolve because of where the temp-dir tests sit. |
| `x_extract_seed_metrics__mutmut_77` | `metrics.get("avg_timesteps")` -> `metrics.get(None)` | `metrics.get(None)` returns `None`, which the `timesteps = timesteps or ...` fallback chain tolerates. |

**Triage:** predominantly equivalent mutants. Optionally harden the `encoding`
cases by asserting the byte content (an explicit UTF-8 multi-byte round-trip);
otherwise keep baselined.

### String-constant content mutation (weak assertion)

mutmut wraps or alters a string constant. The mutation survives because the
tests assert a **substring** match rather than equality:

| Survivor (example) | Mutation | Why it survives |
| --- | --- | --- |
| `x_extract_seed_metrics__mutmut_32` | `raise KeyError("metrics not found")` -> `KeyError("XXmetrics not foundXX")` | `test_extract_seed_metrics_missing_metrics_fails` asserts `"metrics not found" in failure["reason"]`; the wrapped string still contains the substring. |

**Triage:** missing assertion worth adding. Tighten to an exact-equality check
on the reason string (or a `KeyError` message check) to kill this mutant, then
refresh the baseline.

### Logging-only / side-output constant change (equivalent or low-value)

The mutation changes a constant that only affects log fields, not the computed
metric result:

| Survivor (example) | Mutation | Why it survives |
| --- | --- | --- |
| `x_extract_seed_metrics__mutmut_139` | `seed=failure["seed"],` -> `seed=None,` (log kwargs) | The failure `seed`/`policy_type` are read from the failure dict elsewhere; the structured-log kwargs are not asserted by tests. |

**Triage:** mostly equivalent / low-value. Keep baselined unless log-field
correctness becomes contractually required.

### Fallback-chain / optional-field mutations in `extract_seed_metrics`

The largest group (38 survivors) concentrates in `extract_seed_metrics`, which
has a long `metrics.get(...) or metrics.get(...)` fallback chain and many
optional numeric fields. Mutations here often survive because the tests do not
exercise every alias path (`timesteps_to_convergence` / `avg_timesteps` /
`total_timesteps`) or every optional field independently.

**Triage:** mix of missing assertions (worth adding per-alias and per-optional-
field tests) and equivalent mutants (where a fallback branch is genuinely
unreachable for the documented manifest schema). Prioritize assertions for the
aliases that real manifests emit.

## How to reduce the baseline

1. Run the lane locally: `uv run python scripts/dev/mutation_ratchet.py --check`
   (re-runs mutmut; a few minutes on this module) or inspect the latest
   scheduled-run artifact.
2. Pick a baselined survivor and inspect it: `uv run mutmut show <name>`.
3. Classify it as **missing assertion** or **equivalent mutant** using the table
   above.
4. If a missing assertion, add a focused test in
   `tests/research/test_aggregation.py` that fails under the mutant and passes
   on the real source.
5. Re-run `--check` to confirm the mutant is now killed (it will appear as a
   ratchet opportunity), then `--write-baseline` to lock in the reduction.
6. For equivalent mutants, leave them baselined; no action is needed.

## Scope guardrails

- The lane is deliberately limited to `robot_sf/research/aggregation.py`. Do
  not expand the `[mutmut]` `source_paths` in `setup.cfg` to additional modules
  without a separate, scoped decision and a refreshed baseline — the weekly cost
  must stay predictable.
- The lane is a diagnostic, not a PR gate. Do not add it to `pull_request:`
  triggers.
- This is diagnostic-only. It is **not** benchmark evidence, a repository-wide
  mutation score, or a paper-facing result; do not promote it as such.

## Correctness repair found during first-slice gate review

`compute_completeness_score` converted numeric seed values back to strings in
its sort key. That made `"10"` sort before `"2"`. The implementation now returns
`(0, int(value))`, and the regression test asserts numeric-first ordering
(PR #5513).
