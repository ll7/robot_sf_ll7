# Issue #6871 — Report Crosswalk for Diagnosis and Execution Monitoring

## Summary

This context note documents the versioned report/schema crosswalk
(`report_crosswalk.v1`) that integrates deterministic failure-diagnosis and
execution-time deviation monitoring fields into episode and campaign summary
structures for benchmark reporting artifacts.

**Issue**: [#6871](https://github.com/ll7/robot_sf_ll7/issues/6871)
**Module**: `robot_sf/benchmark/report_crosswalk.py`
**Tests**: `tests/benchmark/test_report_crosswalk.py`
**Report source**: `report_crosswalk.v1`

## Problem

The published benchmark follow-up
[#103](https://github.com/ll7/amv_benchmark_paper/issues/103) asks for
failure-centric and execution-time reporting requirements. `robot_sf_ll7` already
has separate upstream work for the underlying diagnostic and monitoring
capabilities; the remaining gap is a reporting crosswalk that makes those outputs
usable in campaign artifacts without changing the meaning of the core success
metrics.

## Existing Upstream Owners

- **[#6583](https://github.com/ll7/robot_sf_ll7/issues/6583)** / merged PR
  [#6625](https://github.com/ll7/robot_sf_ll7/pull/6625): deterministic
  `failure_diagnosis.v1` record and adapter (`robot_sf/benchmark/failure_diagnosis.py`).
- **[#6584](https://github.com/ll7/robot_sf_ll7/issues/6584)** / merged PR
  [#6671](https://github.com/ll7/robot_sf_ll7/pull/6671): execution-time deviation
  monitor (`robot_sf/benchmark/trajectory_verifier.py`, `ExecutionDeviationResult`,
  `monitor_execution_deviation`, `summarize_execution_deviation_diagnostics`).
- **[#6646](https://github.com/ll7/robot_sf_ll7/issues/6646)** / merged PR
  [#6704](https://github.com/ll7/robot_sf_ll7/pull/6704): learned/reference
  diagnosis quality, correction-usefulness scoring, and campaign-level
  diagnosis-quality evaluation.

This crosswalk owns only the reporting contract that surfaces those outputs in
campaign artifacts.  It does not own the underlying diagnostic, monitoring, or
quality-evaluation implementations.

## What the Crosswalk Does

### Maps to Episode Summaries

`build_episode_diagnostic_summary()` maps:

1. **`failure_diagnosis.v1` fields** → diagnosis availability, record counts,
   failure-type counts, severity counts, unknown-type counts, validity state,
   and provenance.
2. **`execution_deviation.v1` fields** → intervention label, deviation score,
   fail-closed flag, threshold crossing time, validity state, and provenance.
3. **Core benchmark metrics** → success, collision, and comfort passed through
   unchanged and kept separate from diagnostic quality.

### Aggregates to Campaign Summaries

`build_campaign_diagnostic_summary()` aggregates:

- Diagnosis coverage rate (episodes with a structurally valid payload / total
  episodes; fallback/degraded payloads remain flagged and are not benchmark
  evidence).
- Execution-deviation coverage rate and fail-closed count.
- Per-intervention-label counts across episodes.
- Success rate, collision rate, and comfort mean (computed independently).

### Backward-Compatible Export

`export_crosswalk_example_fixture()` builds a deterministic example fixture
with four episodes demonstrating:

1. A collision diagnosis (known type).
2. An unknown-type diagnosis (oscillation).
3. A warn execution-deviation case.
4. A fail-closed execution-deviation case.

## Denominators and Validity Rules

| Field | Denominator | Validity States |
|-------|-------------|-----------------|
| `diagnosis_record_count` | Validated records retained by the crosswalk; malformed payloads contribute zero | N/A (integer) |
| `diagnosis_unknown_count` | Records with `failure_type == "unknown"` | N/A (integer) |
| `diagnosis_failure_type_counts` | Per-type record counts | N/A (dict) |
| `diagnosis_coverage_rate` | Episodes with diagnosis / total | `available`, `unavailable` |
| `execution_deviation_coverage_rate` | Episodes with result / total | `available`, `unavailable` |
| `success_rate` | Episodes with known success / total | N/A (bool → rate) |
| `collision_rate` | Episodes with known collision / total | N/A (bool → rate) |
| `comfort_mean` | Episodes with known comfort | N/A (float → mean) |

**Validity states** for per-episode fields:
- `available`: upstream record is present and valid.
- `unavailable`: upstream record is not provided or fail-closed.
- `invalid`: upstream record has invalid provenance or structure.
- `fallback`: value was produced by a fallback path.
- `degraded`: value was produced in degraded mode.

**Provenance states**:
- `complete`: upstream record exists and schema version matches.
- `incomplete`: upstream record exists but schema version differs.
- `unknown`: upstream record was not provided.

## Unavailable / Invalid / Fallback / Degraded Handling

- **Missing diagnosis payload** → `diagnosis_validity_state="unavailable"`,
  `diagnosis_validity_reason="diagnosis_payload_not_provided"`, all counts zero.
- **Wrong diagnosis schema, source, or nested record provenance** →
  `diagnosis_validity_state="invalid"`, `diagnosis_provenance="incomplete"`,
  and no diagnosis denominator or label counts.
- **Missing execution-deviation result** →
  `execution_deviation_validity_state="unavailable"`,
  `execution_deviation_validity_reason="execution_deviation_result_not_provided"`.
- **Fail-closed deviation result** →
  `execution_deviation_validity_state="unavailable"`,
  `execution_deviation_validity_reason="execution_deviation_fail_closed:invalid_or_stale_inputs"`,
  `deviation_score=None` (never fabricated).
- **Fallback/degraded diagnosis records** → the corresponding state is retained
  explicitly; those rows are not promoted to benchmark success, ranking, or
  quality evidence.
- Empty record lists produce zero counts (no forced "unknown" class).

## Claim Boundary

This is a **reporting and artifact-contract improvement**.  It must not turn a
diagnostic record into evidence of:

- Causality
- Safety
- Planner ranking
- Intervention effectiveness
- Generalization
- Benchmark success

### Detection vs. Correction

The crosswalk surfaces execution-deviation detection labels (`continue`, `warn`,
`replan`, `fallback_brake`) as offline diagnostic labels only.  Detecting or
describing an execution deviation is **not the same** as proving that a
controller or policy corrected it.  Correction claims require a separately
approved evaluation.

## Files Changed

| File | Purpose |
|------|---------|
| `robot_sf/benchmark/report_crosswalk.py` | Versioned crosswalk module |
| `tests/benchmark/test_report_crosswalk.py` | Deterministic tests |
| `docs/context/issue_6871_report_crosswalk.md` | This context note |

## Related Issues

- [#6583](https://github.com/ll7/robot_sf_ll7/issues/6583) — failure_diagnosis.v1
- [#6584](https://github.com/ll7/robot_sf_ll7/issues/6584) — execution deviation monitor
- [#6646](https://github.com/ll7/robot_sf_ll7/issues/6646) — learned/reference diagnosis quality
- [#4757](https://github.com/ll7/robot_sf_ll7/issues/4757) — trajectory verifier (predecessor to #6584)
