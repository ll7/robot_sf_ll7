# Issue #1288 JSONL Append Optimization

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1288>

## Goal

`append_jsonl_record` is used by training/runtime logging paths. Before this change it used
standard-library `json.dump(..., sort_keys=True)` for every record, which adds avoidable overhead
for high-throughput JSONL writes.

## Decision

Use `orjson` for JSONL encoding and disable key sorting by default. Preserve legacy deterministic
ordering as an explicit `sort_keys=True` option for callers that need stable sorted output.

The helper now also serializes common runtime logging payload values:

- `pathlib.Path` as strings,
- NumPy arrays as lists,
- NumPy scalar values as native Python scalars.

Nested payload conversion is explicit and iterative before encoding, dictionary keys must be
strings, and non-finite numeric values (`NaN`, `Infinity`) are emitted as JSON `null` values so
standard JSON consumers can parse the JSONL stream.

## Validation

Focused tests:

```bash
./.venv/bin/python -m pytest tests/training/test_runtime_helpers.py -q
```

Result:

```text
10 passed
```

Local write-throughput microbenchmark, 1000 large records with decoded-record parity against
`json.dump(..., sort_keys=True)`:

```text
baseline_json_sort_keys_seconds=0.267688
orjson_append_seconds=0.034867
speedup=7.68x
```

## Follow-Up Boundary

This issue only optimizes the shared JSONL append helper. Broader benchmark-runner export formats,
batch buffering, compression, or Parquet migration should remain separate issues.
