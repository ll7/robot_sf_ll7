# Issue #1721 Benchmark Track Metadata Audit

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1721>

Date: 2026-05-30

## Goal

Audit tracked benchmark evidence summaries, reports, and compact context evidence for explicit
`benchmark_track` and `track_schema_version` metadata now that observation-track identity is part
of benchmark comparability.

This note preserves historical evidence as-is. It does not rerun benchmarks, rewrite old result
files, infer missing metadata, or change planner rankings.

## Classification Terms

- `track_known`: the artifact or source contract explicitly records `benchmark_track` and
  `track_schema_version`, or the note is a current track-aware specification/test surface.
- `legacy_track_unknown`: the artifact contains benchmark results or derived benchmark analysis but
  predates explicit track metadata. It can support bounded historical interpretation, but it must
  not be aggregated with track-aware rows or used for cross-observation-track comparisons.
- `not_applicable`: the artifact is a design note, launch packet, blocked/unavailable row,
  diagnostic stub, CARLA replay parity surface, or non-benchmark evidence where observation-track
  aggregation is not the evidence claim.

## Method

Static audit commands:

```bash
rg "benchmark_track|track_schema_version" docs/context docs/context/evidence configs scripts
rg -l "benchmark_success|campaign_summary|scenario_seed|success_rate|collision|snqi|paper_facing|paired|AMV|seed" \
  docs/context docs/context/evidence -g '*.md' -g '*.json' -g '*.csv' -g '*.yaml'
```

Observed result: no tracked files under `docs/context/evidence/` currently contain
`benchmark_track` or `track_schema_version`. The track-aware contract is present in
`docs/benchmark_spec.md`, `docs/context/issue_1612_observation_track_architecture.md`, current
benchmark tests, and current benchmark/runtime code, but the preserved compact evidence bundles
listed below were produced before that metadata became part of benchmark identity.

## Inventory

| Surface | Evidence kind | Classification | Why | Action |
|---|---|---|---|---|
| `docs/benchmark_spec.md` | current benchmark specification | `track_known` | Defines `benchmark_track` and `track_schema_version` as part of track-aware runs and aggregation boundaries. | Use as the current rule anchor. |
| `docs/context/issue_1612_observation_track_architecture.md` | observation-track architecture note | `track_known` | Defines track metadata fields, aggregation fence, and diagnostic cross-track reporting boundary. | Use as the current interpretation anchor. |
| `tests/test_cli_run.py`, `tests/test_runner_resume.py`, `tests/test_cli_aggregate.py`, `tests/test_cli_table.py`, `tests/test_ranking.py`, `tests/benchmark/test_lidar_observation_track.py` | current track-aware tests | `track_known` | Exercise metadata emission, resume identity, fail-closed mixed-track aggregation, and diagnostic cross-track labels. | Use as validation evidence for new runs, not as historical evidence metadata. |
| `docs/context/evidence/issue_1344_paired_amv_primary_2026-05-20/` | compact AMV primary nominal/stress evidence | `legacy_track_unknown` | Benchmark summaries and tables predate explicit track metadata; static search found no track keys. | Keep cited with AMV/SNQI and unknown-track caveats. |
| `docs/context/evidence/issue_1353_broader_amv_2026-05-26/` | broader AMV nominal/stress/cross-kinematics evidence | `legacy_track_unknown` | Benchmark summaries and tables predate explicit track metadata; static search found no track keys. | Keep cited with AMV/SNQI, SocNav unavailable, and unknown-track caveats. |
| `docs/context/evidence/issue_1454_stage_a_fixed_h100_2026-05-22/` | S10 fixed-h100 benchmark evidence | `legacy_track_unknown` | Benchmark summaries and reports predate explicit track metadata; static search found no track keys. | Use only inside its historical fixed-h100 scope. |
| `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/` | S10/h500 candidate benchmark evidence | `legacy_track_unknown` | Benchmark summaries, seed rows, and campaign tables predate explicit track metadata; static search found no track keys. | Keep challenger and seed-analysis claims bounded to this historical surface. |
| `docs/context/evidence/issue_1462_s10_h500_failure_modes_2026-05-24/` | derived S10/h500 failure-mode summary | `legacy_track_unknown` | Derived from the issue #1454 S10/h500 candidate bundle, so it inherits unknown-track status. | Do not treat as new track-known evidence. |
| `docs/context/evidence/issue_1608_seed_sensitivity_2026-05-30/` | derived seed-sensitivity analysis | `legacy_track_unknown` | Derived from the issue #1454 S10/h500 candidate bundle, so it inherits unknown-track status despite being generated after track architecture work. | Cite as diagnostic prioritization only. |
| `docs/context/evidence/issue_1484_broader_cross_kinematics_2026-05-28/` | compact cross-kinematics smoke/probe evidence | `legacy_track_unknown` | Benchmark summaries and tables predate explicit track metadata; static search found no track keys. | Use only as compatibility smoke/probe evidence. |
| `docs/context/evidence/camera_ready_all_planners_2026-05-04/` and `docs/context/evidence/issue_1023_*_2026-05-06/` | older camera-ready and scenario-horizon compact evidence | `legacy_track_unknown` | Benchmark summaries and reports predate explicit track metadata; static search found no track keys. | Do not mix with track-aware result rows. |
| CARLA replay, route-clearance, setup-smoke, launch-packet, and blocked/unavailable evidence bundles | non-benchmark or non-aggregation evidence | `not_applicable` | These surfaces are not making observation-track benchmark aggregation claims. | Preserve their existing caveats; no track caveat needed unless reused as benchmark rows. |

## Claim Caveat

Any current note that cites the `legacy_track_unknown` benchmark bundles above should include this
boundary:

> Unless a cited evidence row explicitly records `benchmark_track` and `track_schema_version`, treat
> it as `legacy_track_unknown`: useful for within-surface historical interpretation, but not safe to
> aggregate with track-aware results or compare across observation tracks.

This caveat is now reflected in the current manuscript evidence map, seed-budget planning note,
S10/h500 candidate comparison note, and issue #1608 seed-sensitivity note.

## Validation

- `rg "benchmark_track|track_schema_version" docs/context docs/context/evidence configs scripts`
- Manual link and caveat review for:
  - `docs/context/issue_1542_manuscript_claim_evidence_map.md`
  - `docs/context/issue_1545_power_aware_seed_budget_planning.md`
  - `docs/context/issue_1454_s10_h500_candidate_comparison.md`
  - `docs/context/issue_1608_seed_sensitivity_analysis.md`
  - `docs/context/evidence/issue_1608_seed_sensitivity_2026-05-30/README.md`
