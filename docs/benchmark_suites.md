# Benchmark Suites Map

Robot SF has benchmark-suite documentation on this branch, but no separate generated suite catalog.
This page is the docs-site navigation layer over the existing suite, campaign, leaderboard, and
evidence surfaces.

## Suite And Campaign Entry Points

- [Benchmark Spec](benchmark_spec.md) defines the canonical classic-interactions and Francis 2023
  scenario split, seed policy, baseline categories, and reproducible commands.
- [Benchmark Runner And Metrics](benchmark.md) documents the benchmark runner, schema handling,
  metrics, SNQI boundaries, and the local smoke benchmark demo.
- [Camera-Ready Benchmark Workflow](benchmark_camera_ready.md) lists camera-ready, paper-matrix,
  h500, cross-kinematics, SocNavBench, and sanity benchmark configs.
- [Benchmark Release Protocol](benchmark_release_protocol.md) defines release-versioning and
  publication expectations for paper-facing benchmark artifacts.
- [Static Leaderboards](leaderboards/README.md) documents the Markdown leaderboard row contract and
  current smoke, nominal-sanity, AMV, and LiDAR result surfaces.

## Benchmark Boundaries

- [Benchmark Fallback Policy](context/issue_691_benchmark_fallback_policy.md) is the fail-closed
  rule for fallback, degraded, and unavailable rows.
- [Planner Family Coverage](benchmark_planner_family_coverage.md) tracks planner-family coverage
  boundaries.
- [Experimental Planner Guardrails](benchmark_experimental_planners.md) explains when exploratory
  planners are diagnostics rather than benchmark evidence.
- [Evidence Bundles](context/evidence/README.md) describes what belongs in tracked evidence and
  what must remain in disposable `output/` or an external artifact store.

## Config Roots

The source configs remain outside the Sphinx source tree and are linked from the docs above:

- `configs/scenarios/classic_interactions_francis2023.yaml`
- `configs/scenarios/classic_interactions.yaml`
- `configs/scenarios/francis2023.yaml`
- `configs/benchmarks/camera_ready_all_planners.yaml`
- `configs/benchmarks/paper_experiment_matrix_v1.yaml`
- `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml`
