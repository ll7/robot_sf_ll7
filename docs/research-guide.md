# Robot SF — Research & Benchmark Guide

This guide is the **research/benchmark** entry point for Robot SF. It covers benchmark semantics,
evidence grading, scenario certification, release protocol, and the discipline needed to publish a
trustworthy result. It assumes you can already install and run Robot SF (see the
[User Guide](./user-guide.md)); it does not repeat task-oriented setup.

The single most important rule: **do not present diagnostic, smoke, fallback, or degraded output as
benchmark or paper-facing evidence.** Match the claim to the proof.

## 1. Research workflow

- [Researcher's Guide](./researchers_guide.md) — from a research question to a published,
  correctly-graded result.
- [Maintainer Values And Hard Contracts](./maintainer_values.md) — honest, transparent, reproducible
  progress; exploration labels; uncertainty and validation policy.
- [Research Reporting](./research_reporting.md) — how to report research results conservatively.
- [Context Retrieval Index](./context/INDEX.md) — retrieval-first catalog for context-note entry
  points and status rules.

## 2. Benchmark semantics and metrics

- [Benchmark Runner And Metrics](./benchmark.md) — episode schema, aggregation, metrics suite
  (collisions, comfort exposure, SNQI), and validation hooks.
- [Benchmark Spec](./benchmark_spec.md) — formal benchmark specification.
- [Metrics Specification](./dev/issues/social-navigation-benchmark/metrics_spec.md) — formal metric
  definitions, including per-pedestrian force quantiles.
- [Benchmark Suites Map](./benchmark_suites.md) — benchmark suite and campaign surfaces.
- [Full Classic Interaction Benchmark](./benchmark_full_classic.md) — complete episode/aggregation/
  effect-size/plot/video pipeline guide.
- [Curvature Metric](./curvature_metric.md) — curvature-based metric definition.

## 3. Evidence and claim discipline

- [Simulation-Evidence Safety Case Template](./simulation_evidence_safety_case.md) — map benchmark
  artifacts to bounded safety-case sections; name simulation limits and external evidence needs.
- [Code Review Guide](./code_review.md) — benchmark-facing review checklist (semantics,
  normalization, distributions, reproducibility, provenance).
- [Hazard Traceability](./hazard_traceability.md) — intended hazard coverage summary (not safety
  proof).
- [ODD Contracts](./odd_contracts.md) — operating-assumption metadata bounding benchmark evidence.
- [Assurance Fragments](./assurance_fragments.md) — reusable assurance fragments and grading.
- [Artifact Catalog](./artifact_catalog.md) — stable semantic IDs, checksums, and claim boundaries
  for reusable figures and tables.

## 4. Scenario certification and contracts

- [Scenario Certification](./scenario_certification.md) — machine-readable validity, feasibility,
  stress-only, and hard-but-solvable certificates.
- [Scenario Contracts](./scenario_contracts.md) — validate authored scenario-intent contracts.
- [Scenario Perturbation Manifest](./scenario_perturbation_manifest.md) — perturbation coverage.
- [Scenario Thumbnails](./scenario_thumbnails.md) — per-scenario thumbnails and montage grids.
- [Hazard ODD Coverage Rollup](./hazard_odd_coverage_rollup.md) — ODD coverage summary.

## 5. Planner benchmarking and families

- [Benchmark Planner-Family Coverage Matrix](./benchmark_planner_family_coverage.md) — planner/config
  support mapped to Alyassi-style families, with overclaim guardrails.
- [Benchmark: Experimental Planners](./benchmark_experimental_planners.md) — opt-in guardrails for
  unfinished planner families.
- [Benchmark Planner Quality Audit](./benchmark_planner_quality_audit.md) — planner decision table and
  headline suitability classification.
- [Prediction Planner Baseline](./baselines/prediction_planner.md) — model description and provenance.
- [Prediction-Aware MPC Planner](./baselines/prediction_mpc.md) — experimental predictive planner.
- [Dynamic Window Approach Baseline](./baselines/dwa.md) — classical acceleration-window planner.
- [Guarded PPO Baseline](./baselines/guarded_ppo.md) — safety-aware challenger profile.
- [Baselines Overview](./dev/baselines/README.md) — available baseline planners index.

## 6. Release protocol and publication

- [Benchmark Release Protocol](./benchmark_release_protocol.md) — canonical release model, versioning,
  and manifest/entrypoint contract.
- [Benchmark Release Reproducibility](./benchmark_release_reproducibility.md) — reproduce a release
  from a tag.
- [Benchmark Artifact Publication](./benchmark_artifact_publication.md) — public artifact policy,
  DOI-ready export bundles, Zenodo workflow.
- [Release Artifact Badging](./release_artifact_badging.md) — badge/claim boundaries for artifacts.
- [Camera-Ready Benchmark Workflow](./benchmark_camera_ready.md) — full camera-ready campaign guide.
- [Camera-Ready Release Workflow](./benchmark_camera_ready_release.md) — guided release upload
  checklist.
- [Benchmark Docker Reproduction Path](./benchmark_docker_repro.md) — pinned Docker benchmark smoke.
- [Benchmark Observation Visibility](./benchmark_observation_visibility.md) — planner-facing FOV and
  occlusion filtering.
- [Multi-AMV Benchmark](./multi_amv_benchmark.md) — minimal multi-robot scenario surface.

## 7. Reproducibility and provenance

- [Benchmark Campaign Manifest](./benchmark_campaign_manifest.md) — campaign manifest contract.
- [External Repo Setup](./external_repo_setup.md) — pinning upstream planner repos.
- [External Data Setup Assistant](./external_data_setup.md) — dataset provenance manifests.
- [External Planner Reuse Checklist](./context/external_planner_reuse_checklist.md) — fail-fast
  upstream provenance and wrapper smoke.
- [Project Prioritization](./project_prioritization.md) — Project #5 field semantics and score model.

## 8. Leaderboards and governance

- [Static Leaderboards](./leaderboards/README.md) — leaderboard policy and boundaries.
- [Benchmark Governance](./benchmark_governance.md) — governance model and escalation.
- [Benchmark Static Dashboard](./benchmark_static_dashboard.md) — static reporting dashboard.

## Where to go next

- **Need to run something?** See the [User Guide](./user-guide.md).
- **Need to extend the code (planners, scenarios, CI)?** See the [Developer Guide](./developer-guide.md).
