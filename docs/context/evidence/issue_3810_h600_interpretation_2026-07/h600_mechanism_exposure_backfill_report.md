# Issue #4242 h600 mechanism/exposure sidecar backfill

Plain-language summary: this compact sidecar records whether retained h600 episode rows can support trace-verified mechanism and interaction-exposure fields.

Evidence status: diagnostic-only schema closure. This is not a benchmark ranking, paper-facing claim, dissertation claim, or imputation pass.

## Status counts

- Retained episode rows: 2304
- Mechanism statuses: `{"not_derivable_missing_trace": 2304}`
- Interaction-exposure statuses: `{"not_derivable_missing_trace": 2304}`

## Run summary

| job_id | run_label | rows | mechanism statuses | exposure statuses |
| --- | --- | ---: | --- | --- |
| 13268 | confirm | 1008 | `{"not_derivable_missing_trace": 1008}` | `{"not_derivable_missing_trace": 1008}` |
| 13273 | extended_roster | 1296 | `{"not_derivable_missing_trace": 1296}` | `{"not_derivable_missing_trace": 1296}` |

## Claim boundary

Retained h600 sidecars only; rows are trace-derived when retained traces/native fields exist, otherwise explicitly not derivable. No geometry-only mechanism labels, imputation, benchmark ranking, or paper/dissertation claims.
