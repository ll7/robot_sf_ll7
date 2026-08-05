# Issue #4830 safety-wrapper factorial evidence

This directory is a compact, public evidence snapshot for
[`ll7/robot_sf_ll7#4830`](https://github.com/ll7/robot_sf_ll7/issues/4830).
It does not contain the raw episode JSONL files. The raw files remain at the
external campaign result root recorded in the private job ledger.

## Campaign execution

- Slurm job: `13775`
- Private job-ledger campaign: `2026-07-issue4830-safety-wrapper-factorial-v1`
- Public commit: `8b5ce0b1ca5b7845ae05b0fdc07761079c75d380`
- Public configuration: `configs/benchmarks/issue_4830_safety_wrapper_factorial_v1.yaml`
- Public configuration SHA-256: `a507d55282eaf95b8d67c8e2a0764786db9eddc30905531fb5b14da913a97be4`
- Paired seeds: `111`, `112`, `113`
- Arms: three planners × `wrapper_off`/`wrapper_on`
- Episodes: `864` (`144` per arm)
- Episode pairing keys: `432` (`planner`, `scenario_id`, `seed`)
- Completed arms: `6/6`
- Unexpected failed or fallback/degraded arms: `0`
- Slurm exit: `0:0`

The standard camera-ready campaign surface is valid. The copied reports retain
the runner's claim boundary: this is a diagnostic paired campaign, not a safety
certification or dissertation result.

## Paired-report boundary

The existing issue #3501 paired report builder is intentionally not run against
these files as if the contract were satisfied. Its input requires normalized
rows with `metric_values` for every `(planner, scenario_id, seed, wrapper_arm)`.
The camera-ready episode records do not emit that normalized object.

[`audit/summary.json`](audit/summary.json) records the fail-closed result. It
also records fields that are present in the episode records, but it does not
reinterpret `clearing_distance_min`, progress proxies, or wrapper diagnostics
as `min_predicted_separation_m`, `progress_at_timeout`,
`false_positive_stop_rate`, or `stop_yield_latency_s`. The missing semantics
require a reviewed metric contract before a paired effect report or manuscript
evidence admission.

## Contents

- `manifest.json`, `run_meta.json`, and
  `reports/campaign_summary_excerpt.json`: portable execution and provenance
  metadata;
- `preflight.json`: launcher preflight result;
- `reports/`: standard integrity, comparability, matrix, campaign report, and
  credibility artifacts. The raw campaign manifest and full campaign summary
  remain at the external source location in `audit/summary.json` because they
  contain non-portable generated metadata;
- `audit/`: the issue-specific normalized-row contract audit;
- `SHA256SUMS`: checksums for every compact file in this directory.

Do not use this snapshot to change dissertation claims. Evidence admission and
the unresolved paired metric semantics remain separate gates.
