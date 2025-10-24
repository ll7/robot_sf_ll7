# Contract – Episode Metadata Injection (`run_full_benchmark`)

## Overview
Extends the classic benchmark orchestrator so each emitted episode record contains duplicated algorithm metadata.

## Preconditions
- `BenchmarkCLIConfig.algo` is set before invoking `run_full_benchmark`.
- Episode manifests returned by worker processes include a dictionary payload representing the episode JSON prior to serialization.

## Behavioural Requirements
1. Before writing each JSONL line, the orchestrator must:
   - Ensure `record["algo"]` is set to the selected algorithm slug.
   - Mirror this value into `record.setdefault("scenario_params", {})["algo"]`.
2. Validation:
   - If `scenario_params` is not a `dict`, raise `AggregationMetadataError` (same class as aggregation contract) with guidance to regenerate the episode.
   - If `record["algo"]` differs from any pre-existing `scenario_params["algo"]`, overwrite the nested value and log a warning (`logger.warning`) noting the correction.
3. Logging:
   - Emit a `logger.debug` statement per episode with keys `episode_id`, `algo`, `scenario_id` when metadata injection occurs (for reproducibility tracing).
4. Legacy compatibility: when reprocessing an existing JSONL file (resume), skip duplication if both fields already align.

## Outputs
- Episodes written to `episodes/episodes.jsonl` now satisfy the data model documented in `/specs/142-aggregation-mixes-algorithms/data-model.md`.
- Resume manifests produced by `run_full_benchmark` note the enriched metadata (if applicable).

## Error Handling
- On validation failure, raise `AggregationMetadataError` and abort writing to avoid corrupting JSONL. The caller (benchmark script) should surface a fail-fast error per FR-004.

## Instrumentation
- Add structured Loguru context `event="episode_metadata_injection"` for traceability.

## Tests to Author
| Test | Description |
|------|-------------|
| `test_injects_nested_algo_metadata` | Creates a fake manifest entry lacking `scenario_params["algo"]` and asserts the orchestrator writes both fields. |
| `test_raises_on_missing_algo` | Verifies an exception when the top-level `algo` is absent. |
| `test_logs_warning_on_mismatch` | Pre-populates conflicting nested value and asserts warning plus corrected result. |
