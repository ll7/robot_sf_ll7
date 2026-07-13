# Mutation-testing triage for metric aggregation

This report records a bounded mutation diagnostic for `robot_sf/research/aggregation.py`.
It checks whether the selected aggregation tests detect small implementation changes; it is
diagnostic-only and is not benchmark evidence or a paper-facing result.

## Reproduction

From the repository root, with the development dependencies installed:

```bash
uv run mutmut run --max-children 8
uv run mutmut results
```

The run used the `[mutmut]` configuration in `setup.cfg`:

- source: `robot_sf/research/aggregation.py`;
- copied runtime package: `robot_sf`;
- selected tests: `tests/research/test_aggregation.py`.

## Result

| Classification | Count | Interpretation |
| --- | ---: | --- |
| Killed | 200 | Selected tests detected the mutation. |
| Survived | 5 | Reviewed below; each is equivalent under the current contract. |
| No tests | 218 | The selected test file does not exercise the mutated function. |
| Timeout or suspicious | 0 | No execution anomaly was observed. |
| Total | 423 | All generated mutants in this run. |

Among executable mutants (205 killed or survived), the observed kill ratio is 200/205 (97.6%).
The 218 no-test mutants are excluded from that ratio and must not be described as killed.

## Correctness repair found during gate review

`compute_completeness_score` converted numeric seed values back to strings in its sort key.
That made `"10"` sort before `"2"`. The implementation now returns `(0, int(value))`, and
the regression test asserts numeric-first ordering.

## Surviving-mutant dispositions

| Mutants | Disposition |
| --- | --- |
| `aggregate_metrics` 18 and 19 | Equivalent for the current schema: `policy_type` is the grouping metadata field and supported values are non-numeric. Revisit if numeric `policy_type` values become supported. |
| `aggregate_metrics` 59; `bootstrap_ci` 13 | Equivalent: omitting `replace=True` uses NumPy's default replacement behavior. |
| `compute_completeness_score` 3 | Equivalent: changing the non-numeric sort priority from `1` to `2` does not change ordering because numeric keys use `0` and non-numeric keys share the same priority. |

## Untested functions

The 218 `no tests` mutants from the first slice are a coverage gap, not successful mutation
evidence. The second slice (this update) added 12 targeted tests in
`tests/research/test_aggregation.py` covering `export_metrics_json`, `export_metrics_csv`,
`_load_manifest_payload`, and `extract_seed_metrics`, including round-trips, parent-directory
creation, JSONL last-line semantics, empty/malformed manifests, missing metrics, non-numeric
metrics, and missing files. Re-running `mutmut` on the selected test file is expected to move the
majority of those 218 mutants out of the no-test category.

- `export_metrics_json`: 29;
- `export_metrics_csv`: 17;
- `_load_manifest_payload`: 14;
- `extract_seed_metrics`: 158.

The scheduled diagnostic job and any future mutation-score claim must still report the no-test
category separately. Remaining on #5508: a scheduled (weekly/nightly) `mutmut` workflow that
produces a surviving-mutant report and alerts on regressions.
