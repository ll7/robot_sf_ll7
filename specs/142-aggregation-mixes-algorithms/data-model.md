# Data Model – Preserve Algorithm Separation in Benchmark Aggregation

## Episode Record Schema (JSONL)
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `episode_id` | string | ✅ | Unique identifier for the episode run. |
| `scenario_id` | string | ✅ | Identifier for the scenario configuration. |
| `scenario_params` | object | ✅ | Scenario configuration parameters serialized as JSON. Must now include `scenario_params["algo"]`. |
| `scenario_params["algo"]` | string | ✅ | Algorithm identifier mirrored from the top-level `algo`. Populated when writing the record; absence should trigger validation failure for newly generated episodes. |
| `algo` | string | ✅ | Canonical algorithm label (`sf`, `ppo`, `random`, etc.). Serves as authoritative fallback for legacy files where `scenario_params["algo"]` is missing. |
| `seed` | integer | ✅ | Deterministic seed for reproducibility. |
| `metrics` | object | ✅ | Social navigation metrics (collisions, success rate, SNQI, etc.). |
| `metrics.snqi` | number | ⚠️ | Optional – recomputed on the fly if weights/baseline supplied. |
| `status` | string | ✅ | Episode completion status (`completed`, `failed`, `aborted`). |
| `timestamps` | object | ⚠️ | Optional – start/end timestamps for benchmarking bookkeeping. |
| `manifest_path` | string | ⚠️ | Optional – path to manifest entry for resume support. |

### Validation Rules
- When creating new JSONL entries, `scenario_params["algo"]` **must** match the top-level `algo`. Any divergence is a data integrity error.
- Legacy episodes missing the nested key are still ingestible, but aggregation must log a warning and treat the top-level `algo` as the grouping key.
- Records lacking either `algo` or `scenario_id` are considered invalid and should trigger fail-fast validation before aggregation proceeds.

## Aggregation Output Structure (`compute_aggregates_with_ci`)
```
{
  "sf": {
    "collisions": {"mean": 0.01, "median": 0.0, "p95": 0.05, "mean_ci": [0.0, 0.02], ...},
    "snqi": {...}
  },
  "ppo": {...},
  "_meta": {
    "group_by": "scenario_params.algo",
    "effective_group_key": "scenario_params.algo | algo",
    "missing_algorithms": ["random"],
    "warnings": ["Missing algorithms detected: random"]
  }
}
```

### Output Fields
- **Group buckets** (`sf`, `ppo`, `random`, ...): Dict keyed by algorithm identifier with metric aggregates (mean/median/p95) plus optional confidence intervals (`mean_ci`, `median_ci`, `p95_ci`).
- **`_meta.group_by`**: Records the configured grouping path to aid debugging.
- **`_meta.effective_group_key`**: Documents the fallback chain used (`scenario_params.algo` first, then `algo`, then `scenario_id`).
- **`_meta.missing_algorithms`**: List of expected algorithms that were not present in the aggregated episodes.
- **`_meta.warnings`**: Human-readable log summary repeated inside the JSON output to complement Loguru warnings.

### Validation Rules
- Aggregation must compute `missing_algorithms` by comparing detected algorithm buckets against the set requested by the benchmark manifest (available via orchestrator output or CLI context).
- If `missing_algorithms` is non-empty, the JSON output must still succeed but emit corresponding Loguru warnings.
- `_meta` is mandatory when warnings are emitted; optional otherwise but recommended for traceability.

## Diagnostic Structures
- **Warning Event**: when missing algorithms detected, a structured log entry should include keys `event="aggregation_missing_algorithms"`, `expected`, `present`, `missing` to maintain observability.
- **Validation Error Payload**: when required metadata is absent (e.g., missing `algo` entirely), raise a custom exception with attributes `episode_id`, `missing_fields`, `advice` so calling scripts can halt with actionable messages.

## Relationships & Lifecycle
1. **Episode generation** (`run_full_benchmark`): produce manifests, emit JSONL lines with mirrored algorithm metadata.
2. **Aggregation** (`compute_aggregates_with_ci`): read JSONL, derive group buckets using `scenario_params.algo → algo → scenario_id` fallback chain, populate `_meta` diagnostics, and warn if algorithms missing.
3. **Reporting** (`aggregated_results.json`): consumed by visualization scripts; they should check `_meta.missing_algorithms` to annotate figures or fail fast if necessary.
