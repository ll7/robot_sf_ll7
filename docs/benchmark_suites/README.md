# Benchmark Suite Catalog

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1863>

This catalog names the smallest useful Robot SF benchmark and diagnostic suites. It is a
navigation layer over existing configs, runners, evidence notes, and policies; it does not promote
any suite to paper-grade evidence.

## Evidence Boundary

Use these policies with every suite in this catalog:

- [Benchmark fallback policy](../context/issue_691_benchmark_fallback_policy.md): fallback,
  degraded, failed, skipped, or unavailable rows are caveats or exclusions, not successful
  benchmark outcomes.
- [Artifact evidence vocabulary](../context/artifact_evidence_vocabulary.md): durable claims need
  tracked compact evidence, release artifacts, or explicit external artifact pointers.
- [Benchmark camera-ready workflow](../benchmark_camera_ready.md): publication-grade campaign
  claims require the camera-ready artifact contract, not only a local smoke run.
- [Static leaderboards](../leaderboards/README.md): leaderboard rows must keep `status`,
  `benchmark_track`, `evidence_uri`, and `claim_boundary` visible.

## Suites

| Suite | Purpose | Status |
| --- | --- | --- |
| [Smoke](smoke.md) | Cheapest policy-search candidate sanity run. | runnable local diagnostic |
| [Nominal Sanity](nominal_sanity.md) | First multi-scenario policy-search gate. | runnable local diagnostic |
| [Stress Slice](stress_slice.md) | Harder local stress stage for policy-search candidates. | runnable local diagnostic |
| [AMV Actuation Diagnostic](amv_actuation_diagnostic.md) | Synthetic AMV actuation stress diagnostic. | runnable local diagnostic, not paper-facing |
| [LiDAR 2D Track](lidar_2d_track.md) | LiDAR observation-track smoke and compatibility surface. | contract/training-smoke surface |
| [Adversarial Smoke](adversarial_smoke.md) | Bounded adversarial route/search smoke. | development stress evidence |

## Suite Field Contract

Each suite page records:

- `suite_id`
- purpose
- scenario IDs or scenario config
- seed set
- eligible planners
- `benchmark_track`
- metrics
- expected runtime
- canonical command
- claim boundary
- fallback/degraded caveats

Keep future additions config-first and repository-relative. If a command writes raw output under
`output/`, treat that output as local and non-durable until a compact tracked evidence bundle or
explicit artifact pointer is created.
