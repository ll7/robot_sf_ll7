# Issue #1163 Manual-Control Recording Format Decision

Related issue: [#1163](https://github.com/ll7/robot_sf_ll7/issues/1163)
Related prerequisite: [#1153](https://github.com/ll7/robot_sf_ll7/issues/1153)
Related PR: [#1173](https://github.com/ll7/robot_sf_ll7/pull/1173)

## Decision

Keep append-only JSONL as the canonical manual-control recording format. Do not add a general compact
recording format yet.

The current evidence does not show that JSONL is too large or too slow for horizon-500
manual-control attempts. The already-landed behavior-cloning export remains the compact derived
dataset path when training samples are needed, and it is derived from schema-validated JSONL records
with source-record provenance.

## Evidence

Issue #1153 added the reusable profiling path:

```bash
uv run python scripts/manual_control/profile_recording.py \
  --input output/manual_control/session.jsonl \
  --output output/manual_control/session_profile.json
```

The retained #1153 PR evidence proves the profiler, replay grouping, and BC export helpers, but it
does not include a durable human-run horizon-500 recording profile or write-throughput artifact.
That absence matters: it means a new storage backend would be an assumption-driven change rather
than a measured fix.

As a bounded local proof for this decision, a schema-valid horizon-500 fixture was generated on
2026-05-14 through `ManualJsonlRecorder` and read back through
`profile_manual_jsonl_recording(...)`. The fixture used 500 training-sample records, one attempt,
mapped actions, session metadata, metrics, and a moderate structured observation payload.

Observed fixture profile:

| Metric | Value |
| --- | ---: |
| Records | 500 |
| Attempts | 1 |
| Size | 604,890 bytes |
| Estimated horizon-500 size | 604,890 bytes |
| Bytes per record | 1,209.78 |
| Write time | 0.0236 seconds |
| Write throughput | 21,160 records/second |
| Read time | 0.0071 seconds |
| Read throughput | 70,339 records/second |

This is not a human-control benchmark result, and it should not be used for paper-facing runtime
claims. It is enough to decide that the current schema and profiler do not justify adding a
second canonical recording format.

## Thresholds

Treat JSONL as acceptable while representative manual-control recordings stay within all of these
bounds:

- estimated horizon-500 size is at or below 10 MiB per attempt,
- sustained write throughput is at least 1,000 records/second on the collection host,
- read throughput is at least 1,000 records/second for validation/export workflows,
- the recorder keeps source session, scenario, seed, attempt, step, mode, observation, action, and
  event metadata inspectable without a custom binary reader.

Open a compact-format follow-up only when a representative human-run or runner-generated profile
violates one of those thresholds, or when image-heavy observation capture changes the payload size
class materially.

## Compact Options Compared

Compressed JSONL is the lowest-friction follow-up because it preserves line-oriented records and can
retain the current schema with minimal tooling. It should be the first option if size alone becomes
a problem.

Parquet or Arrow may be useful for batch analytics, but nested observation dictionaries and event
records make the schema less transparent. Use it only for derived datasets, not as the source of
truth.

NPZ is appropriate for homogeneous tensor batches, especially behavior-cloning samples, but it is a
poor fit for mixed event streams with pause, retry, terminal, and metadata records.

A custom binary format is not justified without much larger recordings or a measured write-path
blocker. It would make validation and schema migration harder.

## Provenance Requirement

Any future compact artifact must be derived from JSONL and carry enough provenance to recover the
source:

- source recording path or durable artifact pointer,
- source record schema and source record index,
- session id and manual-control mode metadata,
- scenario id, seed, attempt id, and step index,
- export schema/version and creation command.

The existing `manual_control_bc_v1` export already follows this direction for behavior-cloning
samples and should remain the narrow compact path until broader profiling says otherwise.

## Validation

Validation for this decision:

```bash
uv run pytest tests/docs/test_manual_control_recording_format_decision.py -q
```

Before changing this policy, rerun the profiler on retained representative recordings and record
the profile summary in this note or a linked evidence bundle.
