# Issue #2919 Scenario Prior Gap Report

- Evidence status: `analysis_only`.
- Claim boundary: analysis_only_repository_prior_gap; no planner ranking, benchmark superiority, real-world representativeness, or dataset-backed prior claim.
- Registry input: `configs/research/scenario_prior_cards_issue_2917.yaml`.
- Dataset-backed SDD/ETH/AMV scenario-prior comparison is explicitly deferred to #3161.
- No planner ranking, benchmark superiority, or real-world representativeness is inferred.

## Method

The script loads the Issue #2917 prior-card registry, selects cards classified as `authored` or `repository_trace_derived`, and extracts numeric samples only from machine-readable YAML/JSON source traces. Text docs, Python scripts, raw datasets, and external-dataset candidate cards are excluded from this run.

Distances are diagnostic: KS distance uses empirical CDF separation and the Wasserstein-like value averages absolute matched-quantile gaps over a 0-100% grid.

## Parameter Comparisons

| Parameter | Class | Authored range | Trace-derived range | KS | Wasserstein-like | Proposal |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| pedestrian_density | `too_broad` | 0.02 to 0.12 (n=8) | 0.02 to 0.02 (n=1) | 0.75 | 0.040095 | Split authored `pedestrian_density` stress variants into a centered trace-aligned family near median 0.02 plus a separately labeled stress/extreme family. |
| pedestrian_speed | `too_broad` | -0.25 to 2 (n=25) | 0.8 to 1.4 (n=2) | 0.64 | 0.623267 | Split authored `pedestrian_speed` stress variants into a centered trace-aligned family near median 1.1 plus a separately labeled stress/extreme family. |
| timing_offset_s | `too_extreme` | -0.5 to 1 (n=28) | 0 to 2 (n=4) | 0.5 | 0.649208 | Add a `timing_offset_s_centered_trace_probe` family around trace median 1.0 and keep out-of-support authored variants labeled diagnostic stress only. |

## Classification Notes

- `too_narrow`: authored support is materially inside the trace-derived support.
- `too_broad`: authored support is materially wider than trace-derived support.
- `too_extreme`: authored support is shifted outside trace-derived central mass or support.
- `representative`: authored and trace-derived ranges broadly overlap for this repository-only comparison.

## Limitations

- This is not dataset-backed prior calibration; SDD/ETH/AMV comparison is deferred to #3161.
- Repository trace-derived values come from existing repo configs and compact summaries, not raw real-world traces.
- Some canonical groups mix offsets, caps, and realized values; proposals are scenario-family design prompts, not statistical claims.
- No planner ranking, benchmark superiority, or real-world representativeness is inferred.
