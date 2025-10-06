# Contract – Algorithm-Aware Aggregation (`compute_aggregates_with_ci`)

## Overview
Ensures benchmark aggregation respects per-algorithm separation by enriching records and applying a deterministic grouping fallback chain.

## Inputs
- `records: list[dict]`
  - Each record must include `algo` (string) and `scenario_params` (dict).
  - `scenario_params["algo"]` must match `algo` for newly generated episodes.
- Keyword arguments:
  - `group_by` default: `"scenario_params.algo"`
  - `fallback_group_by` default: `"scenario_id"`
  - `bootstrap_samples`: integer ≥0
  - `bootstrap_confidence`: float (0,1]

## Behaviour
1. When `scenario_params.algo` exists, it is used as the grouping key.
2. If the nested key is missing, the function reads the top-level `algo` field before falling back to `fallback_group_by`.
3. For each group, outputs metric aggregates (`mean`, `median`, `p95`) and optional bootstrap CIs (`*_ci`).
4. Adds `_meta` dictionary containing:
   - `group_by`
   - `effective_group_key`: string denoting the fallback chain actually used (`"scenario_params.algo | algo | scenario_id"`)
   - `missing_algorithms`: sorted list of algorithms expected but absent (derived from optional `expected_algorithms` context when provided – see extension below)
   - `warnings`: list of human-readable diagnostics mirroring emitted Loguru warnings.
5. Emits a Loguru warning with event name `aggregation_missing_algorithms` when `missing_algorithms` is non-empty.

## Error Handling
- Raises `AggregationMetadataError` (new custom exception) when **both** `scenario_params.algo` and `algo` are missing.
- Raises `ValueError` if `bootstrap_samples < 0` or `bootstrap_confidence` outside (0,1].

## Extension Hooks
- Optional parameter `expected_algorithms: set[str] | None` (default `None`): when provided, the implementation compares detected algorithms against this set for diagnostic reporting.
- Optional parameter `logger_ctx: loguru.Logger | None` to support dependency injection in tests.

## Test Matrix (to be implemented)
| Scenario | Expected Outcome |
|----------|------------------|
| New episodes with mirrored metadata | Aggregates grouped strictly by nested key; `_meta.missing_algorithms == []`; no warnings. |
| Legacy episodes lacking nested key | Aggregates grouped by top-level `algo`; `_meta.effective_group_key` reflects fallback. |
| Episodes missing both keys | `AggregationMetadataError` raised before aggregation. |
| Missing baseline vs expected set | Warning emitted, `_meta.missing_algorithms` populated, aggregation succeeds. |
| Bootstrap disabled | Output omits `*_ci` keys, still returns `_meta`. |

## Dependencies
- `robot_sf.benchmark.aggregate` module (existing functions).
- New custom exception class placed under `robot_sf.benchmark.errors` (or similar shared location) with docstring per Constitution Principle XI.
