# Issue #3216 — Headline 7x7 CI + Rank-Stability Report Harness

**Status**: local report harness + small-seed smoke + SLURM launch packet.
The paper-grade run is SLURM and out of local scope. This harness never
self-certifies paper-grade.

## What this delivers

The 7x7 headline planner comparison (planner x scenario-family cells, multi-seed)
was reported with bare point estimates. Issue #3216 attaches **statistical
support**: per-cell confidence intervals and a **rank-stability** statistic
(bootstrap rank distribution / Kendall-tau + rank-flip rate across seed
resamples), so close rankings / safety deltas are reported with uncertainty.

The local deliverable is the **report harness that consumes headline-comparison
rows**, a **small-seed smoke** proving it runs and classifies conservatively,
and the **SLURM launch packet** for the increased-seed-budget run.

### Harness

`scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py`

- Loads headline rows via `canonical_table_export.load_rows_json`
  (`--rows` or `--campaign`).
- Supports `--dry-run` for a deterministic local preflight that exercises the
  same report builder without campaign artifacts or Slurm submission; the fixture
  remains diagnostic-only.
- **Per cell**: point estimate + bootstrap/per-seed confidence interval on the
  primary metrics, with fail-closed cell status — `degraded` / `fallback` /
  `not_available` execution modes and non-success row statuses are excluded and
  never counted as success, with an explicit `exclusion_reason`.
- **Rank stability**: per-scenario planner ordering + Kendall-tau / rank-flip
  across seed resamples; reports whether the headline ranking is stable under
  resampling (`kendall_tau_mean`, `kendall_tau_min`, `rank_flip_rate`,
  `top1_stable`).
- Emits `result.json` + `report.md` and classifies the result
  `paper_grade | nominal | diagnostic | blocked_until_run`. **Insufficient seed
  budget (S10 or fewer) is never paper-grade**: it emits `diagnostic`; an all-
  excluded input emits `blocked_until_run`. Even at S20+ per cell the harness
  emits `blocked_until_run` because paper-grade promotion requires the actual
  predeclared S20/S30 SLURM run (#1554) plus claim-card review. Captures git HEAD.

### Launch packet

`configs/benchmarks/headline_ci_rank_stability_issue_3216_launch_packet.yaml`

References the real canonical configs with verified sha256:

- `paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml`
- `paper_experiment_matrix_7planners_v1.yaml`
- `seed_sets_v1.yaml`, `classic_interactions_francis2023.yaml`,
  `scenario_horizons_h500.yaml`

S20 (`paper_eval_s20`) and the launch-only S30 schedule (`paper_eval_s30`) are
available via #1554. No S20/S30 campaign rows exist yet, so the report remains
blocked until the SLURM run produces durable rows.

## Reuse of canonical owners (no reinvention)

Per AGENTS.md "Canonical Owner Check", the statistics are **composed**, not
re-implemented:

- `robot_sf.benchmark.seed_variance._stats_for_vals` / `_bootstrap_mean_ci` —
  per-cell bootstrap CI over per-seed means.
- `robot_sf.benchmark.fidelity_rank_stability.rank_planners` / `kendall_tau` /
  `count_rank_flips` — per-scenario rank stability across seed resamples.
- `robot_sf.benchmark.canonical_table_export.load_rows_json` — input rows.

No second bootstrap / rank / Kendall implementation was added.

## Coordination

- **#1554** (S20/S30 per-seed bundle, PR #3506): supplies the S20/S30 seed
  schedules and the increased-seed-budget headline campaign. #3216 **consumes**
  the resulting rows; it does not own the seed schedule or campaign execution.
- **#3078** (Package A rank-stability + held-out-family transfer campaign):
  a broader seed/planner-rank stability + transfer campaign. #3216 is the
  **distinct grid-level per-cell** (planner x scenario) headline CI/rank-stability
  report. They share canonical owners (`fidelity_rank_stability`,
  `seed_variance`) rather than duplicating logic.

## Claim boundary

Per-cell CIs and rank-stability are reported with explicit fail-closed cell
status. This harness makes **no paper-grade or planner-ranking claim** on its
own: the paper-grade 7x7 headline run requires the increased seed budget
(S20/S30 via #1554) and is SLURM. On insufficient seed budget the result is
classified `blocked_until_run` or `diagnostic`. Do not fabricate paper-grade
numbers; do not make significance claims from S10/S3.
