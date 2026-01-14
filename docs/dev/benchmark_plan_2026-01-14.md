# Local Navigation Benchmark Gap Analysis (2026-01-14)

Purpose: Provide a grounded, evidence-backed issue description of what is missing to build a robust benchmark that compares local navigation policies and algorithms in this repo. This document inventories what exists, highlights gaps, and frames the open questions and a proposed structure for the work.

## Executive summary

We already have a strong foundation: a benchmark CLI with JSONL outputs, a rich metrics implementation, map-based scenarios, and baseline planners. However, the current system is split across two pipelines (synthetic FastPysf-only vs. Gym environment + map scenarios) and does not yet provide a unified, reproducible benchmark for comparing local navigation policies. The key missing pieces are: (1) a single canonical scenario schema and runner that executes map-based scenarios with real local planners, (2) a finalized baseline algorithm set (beyond Social Force / PPO / Random), (3) data plumbing for obstacle-aware metrics and shortest-path efficiency, and (4) a reproducible evaluation protocol (seeds, sample size, CI targets, and SNQI baseline stats).

The remainder of this document details evidence and a structured gap analysis, then proposes a phased approach with explicit decisions to unblock implementation.

## Current state inventory (evidence and assets)

### 1) Benchmark runners and schemas

- `robot_sf/benchmark/runner.py` runs a synthetic benchmark using `robot_sf/benchmark/scenario_generator.py` and `FastPysfWrapper`. It outputs per-episode JSON records with `scenario_params`, `algorithm_metadata`, and metrics. This path does not use map-based Gym environments or the SVG scenario packs.
- `robot_sf/benchmark/full_classic/` and `scripts/classic_benchmark_full.py` run the map-based "classic interactions" suite with the Gym environment. The orchestrator currently uses a simple goal-seeking policy for rollouts, not a baseline/planner integration.
- Canonical schema under `robot_sf/benchmark/schemas/episode.schema.v1.json` is smaller than the schema in `docs/dev/issues/social-navigation-benchmark/episode_schema.json`; the runner emits keys (`scenario_params`, `algorithm_metadata`, `timestamps`, `config_hash`) that are only defined in the docs-side schema.
- Scenario schema under `robot_sf/benchmark/schema/scenarios.schema.json` covers the synthetic parameterized scenario generator (density, flow, obstacle) and does not validate the map-based scenario YAMLs in `configs/scenarios/`.

### 2) Metrics

- `robot_sf/benchmark/metrics.py` implements core navigation, comfort/force, smoothness/energy, and paper-style metrics (collisions, near misses, jerk, curvature, wall collisions, time to goal, etc). Metrics are documented in `docs/dev/issues/social-navigation-benchmark/metrics_spec.md`.
- Several metrics depend on optional data that is currently not provided by the runner or full-classic pipeline:
  - Wall and obstacle collision metrics require `EpisodeData.obstacles` (currently never populated).
  - Shortest-path efficiency falls back to Euclidean distance; map-aware shortest-path length is not computed.
  - Force field gradient metrics require `force_field_grid` sampling, which is not produced.

### 3) Baselines and local policy adapters

- Bench baseline registry in `robot_sf/baselines/` exposes `social_force`, `ppo`, and `random` baselines (plus simple policy inside the runner).
- `robot_sf/planner/socnav.py` includes lightweight SocNav-style adapters (Sampling, Social Force, ORCA-like, SA-CADRL-like) and a fallback adapter that uses upstream SocNavBench if `output/SocNavBench` is available.
- These SocNav adapters are used in demo scripts (`examples/advanced/18_socnav_structured_observation.py`, `examples/advanced/19_planner_visual_comparison.py`, `examples/advanced/31_classic_planner_orca_demo.py`) but are not wired into `robot_sf_bench` or the full-classic benchmark pipeline.
- The ORCA baseline decision is explicitly deferred (`docs/dev/issues/social-navigation-benchmark/adding_orca.md`).

### 4) Scenarios and maps

- Map-based scenario suites are defined in `configs/scenarios/classic_interactions.yaml` and `configs/scenarios/francis2023.yaml` with SVG maps under `maps/svg_maps/`.
- Francis 2023 SVG maps are geometry-only and explicitly note that per-pedestrian speed tuning and behavior logic (wait/join/leave/gesture) are pending (`maps/svg_maps/francis2023/readme.md`).
- Synthetic scenario generator remains in use by `robot_sf/benchmark/runner.py` (simple rectangular area and simple obstacle layouts).

### 5) Outputs and examples

- Benchmark examples exist (`examples/benchmarks/*`), including a full classic benchmark demo and SNQI flow demo.
- Output artifacts live under `output/` with examples of benchmark outputs in `output/benchmarks/`.

### 6) Reference projects checked locally

- `output/SocNavBench` clone includes a joystick API, synchronous/asynchronous simulator interface, and metrics utilities. The upstream sampling planner can be used via `SocNavBenchSamplingAdapter`.
- `output/arena-rosnav` clone includes the task generator and scenario task definitions; the benchmark mode config and task generator illustrate how ROS-based local planners are benchmarked in a standardized pipeline.

## External reference benchmarks (web sources)

These are the closest analogs to the desired benchmark and provide patterns worth emulating:

- SocNavBench: a grounded simulation testing framework for evaluating social navigation with curated scenarios derived from real-world pedestrian datasets. Source: SocNavBench paper (Biswas et al., arXiv:2103.00047).
- DynaBARN: a dynamic obstacle benchmark that provides 60 Gazebo environments, difficulty metrics, and baseline results for DWA, TEB, RL, and behavior cloning planners; supports different obstacle motion profiles. Source: DynaBARN project page and abstract.
- BARN (static): a benchmark suite of 300 generated navigation environments ordered by difficulty metrics, with baseline DWA and Elastic Bands results, focused on the full sense-plan-act pipeline. Source: BARN dataset page and Benchmarking Metric Ground Navigation abstract.
- Arena-Rosnav: a ROS 2 benchmarking infrastructure with standardized task generator and benchmark mode configurations; uses scenario files for evaluation and a fixed evaluation pipeline. Source: Arena-Rosnav docs and benchmark mode docs.
- MRPB / Local Planning Benchmark: a benchmark of local planners with complex environments and dynamic pedestrians, focusing on local planner comparisons. Source: local-planning-benchmark project page.

## Gap analysis: what is missing and why it matters

The gaps below are organized by impact on the benchmark goal (comparing local planners fairly and reproducibly).

### A) Benchmark scope and definition is not finalized

Evidence:
- Two competing scenario representations: synthetic scenario schema vs. map-based YAML suites.
- Full classic benchmark focuses on a goal-seeking policy and does not yet run baseline algorithms.

Why it matters:
- Without a clear definition of "local navigation policy" vs. "global planning" responsibilities, comparisons will not be apples-to-apples. DynaBARN and BARN explicitly standardize robot size, sensing, and evaluation rules, which reduces ambiguity.

Missing decisions:
- Are we benchmarking local planners with a fixed global route (e.g., visibility graph), or full navigation stacks?
- Which scenario suite is canonical: classic interactions, Francis 2023, synthetic matrix, or a merged set?

### B) Pipeline mismatch: synthetic runner vs. map-based Gym environment

Evidence:
- `robot_sf/benchmark/runner.py` uses `generate_scenario` (synthetic) and does not load SVG maps.
- `robot_sf/benchmark/full_classic/orchestrator.py` uses map-based Gym env but only a simple goal policy.

Why it matters:
- Local planner policies typically need a full environment (obstacles, routes, occupancy grids). The current benchmarking runner does not exercise those inputs, and the map-based benchmark is not yet wired to baseline planners.

Missing work:
- Decide on a single canonical runner and integrate baseline policies into it.
- Ensure the runner exposes observation formats suitable for local planners (structured SocNav + occupancy grid).

### C) Scenario schema and validation do not cover map-based scenario YAMLs

Evidence:
- `robot_sf/benchmark/schema/scenarios.schema.json` only validates the synthetic scenario format.
- `configs/scenarios/classic_interactions.yaml` and `configs/scenarios/francis2023.yaml` use a different schema (map_file, simulation_config, single_pedestrians, metadata).

Why it matters:
- Without schema validation for the map-based scenarios, we cannot guarantee repeatability or catch schema drift.

Missing work:
- Define a canonical scenario schema that covers map-based scenario fields, or version two schemas with explicit selection rules.

### D) Baseline algorithm set is incomplete and not aligned with local planning benchmarks

Evidence:
- Baseline registry includes Social Force, PPO, Random. ORCA is deferred.
- SocNav-style local planners (Sampling, ORCA-like, SA-CADRL-like) are only available via adapters and not part of the benchmark runner.
- DynaBARN and BARN both include classical local planners (DWA, TEB/Elastic Bands) as baselines.

Why it matters:
- A local navigation benchmark should compare local planners that are common in the literature (DWA, TEB, ORCA/RVO). Without them, benchmark results may not be persuasive or comparable.

Missing work:
- Decide which baselines are required for the first benchmark release.
- Integrate at least one reciprocal-avoidance baseline (ORCA/RVO) and one optimization-based local planner (TEB/DWA) if feasible.

### E) Observation/action interface for local planners is not standardized

Evidence:
- SocNav structured observations exist, but benchmark runner uses synthetic numeric policies without standard observation objects.
- PPO adapter uses custom vector or image observation conversion; risks mismatch with local planner inputs.

Why it matters:
- If observation/action interfaces differ across planners, comparisons will be confounded. DynaBARN/BARN explicitly standardize robot size, sensors, and evaluation rules to avoid this.

Missing work:
- Define a single local-planner observation contract (e.g., structured SocNav + occupancy grid) and adapt all baselines to it.
- Define fixed action limits (v_max, omega_max) and ensure these are applied consistently.

### F) Metrics are implemented but key data inputs are missing

Evidence:
- Wall/obstacle collisions and clearing distances are computed in `metrics.py` but `EpisodeData.obstacles` is never populated.
- Shortest path efficiency currently uses Euclidean distance instead of map-aware shortest path length.
- Force gradient metrics require pre-sampled force field grids, not currently produced.

Why it matters:
- A benchmark with incomplete metrics risks misleading comparisons. DynaBARN and BARN emphasize difficulty metrics and repeatable measurement to make results defensible.

Missing work:
- Populate obstacle lists from SVG map geometry when computing metrics.
- Compute map-based shortest path for path efficiency (e.g., via existing planners).
- Decide whether force-field-gradient metrics are required or optional for the first benchmark release.

### G) Reproducibility and provenance are not unified

Evidence:
- Two different episode schemas exist; the canonical schema does not include some fields the runner emits.
- PPO baseline may silently fall back to a goal-seeking policy when the model file is missing.

Why it matters:
- Reproducibility is a core promise of the benchmark (see repo constitution). Missing or ambiguous metadata reduces trust in results.

Missing work:
- Decide which schema is canonical and update all outputs to match it.
- Record algorithm loading status (e.g., "PPO fallback used") in `algorithm_metadata`.

### H) Scenario behavior semantics are incomplete

Evidence:
- Francis 2023 maps note missing speed tuning and behavior logic for overtaking, joining, waiting, etc.
- Map-based crowd behavior relies on ped density and routes, but not all scenario behaviors are implemented in the simulator.

Why it matters:
- Without behavior semantics, the scenarios may not reflect the intended interaction archetypes and comparisons across planners may be invalid.

Missing work:
- Implement the behavior hooks for single-ped scenarios (speed, wait, join/leave).
- Validate that each scenario achieves its intended interaction pattern.

### I) Evaluation protocol is not yet codified

Evidence:
- There are metrics and aggregation utilities, but no finalized benchmark protocol (seed counts, CI targets, acceptance criteria).
- The full classic benchmark includes adaptive precision thresholds but is not yet tied to local planner comparisons.

Why it matters:
- To compare local planners, we need a shared evaluation protocol (seeds, target CIs, and pass/fail criteria) similar to what BARN and DynaBARN provide.

Missing work:
- Define seed counts, CI thresholds, and minimal scenario coverage for benchmark runs.
- Decide whether adaptive sampling is required for all runs or only for full evaluation.

## Open questions (decisions needed before implementation)

1) Which pipeline is authoritative for the benchmark: synthetic runner, full classic benchmark, or a merged runner?
2) What is the definition of "local navigation policy" in this benchmark (local planner only vs. full navigation stack)?
3) Which scenario suites are required for the first release (classic interactions, Francis 2023, synthetic matrix, or a merged set)?
4) Which baseline algorithms must be included (SF, PPO, Random + ORCA/RVO + DWA/TEB?) and what is the inclusion bar?
5) What observation/action contract will all baselines share (structured SocNav, occupancy grid, LiDAR-like features)?
6) Which metrics are mandatory vs. optional (especially obstacle/wall collisions and force-field-gradient metrics)?
7) How do we compute shortest-path efficiency on map-based scenarios (planner-based path vs. grid shortest path)?
8) What is the minimal seed count and CI targets for publishing benchmark results?
9) How should model artifacts (PPO, imitation) be versioned and validated?
10) Do we need compatibility layers to compare with external benchmarks (SocNavBench joystick API, DynaBARN/BARN formats)?

## Proposed structure for the benchmark effort (phased)

### Phase 0: Decision record (scope + schema)
- Choose canonical benchmark pipeline and scenario schema.
- Fix the baseline algorithm set for v1.0 and document inclusion/exclusion rationale.
- Finalize the local planner observation/action contract.

### Phase 1: Unify runner and scenario schema
- Implement a single runner that can load map-based scenarios and execute local planners.
- Add schema validation for map-based scenario YAMLs.

### Phase 2: Baseline integration and metadata
- Wire SocNav-style planners (Sampling, Social Force, ORCA-like, SA-CADRL-like) into the benchmark registry.
- Add at least one classical local planner baseline (DWA or TEB) if feasible.
- Record detailed `algorithm_metadata` (config hash, model path, fallback flags).

### Phase 3: Metrics plumbing
- Populate obstacle data from SVG map geometry to enable wall collisions and clearing distance metrics.
- Compute map-aware shortest path for path efficiency.
- Decide if force-field gradient metrics are in or out for v1.

### Phase 4: Protocol and reporting
- Define the evaluation protocol: seeds, CI thresholds, success criteria.
- Produce baseline stats for SNQI normalization.
- Lock the minimal scenario suite and publish benchmark artifacts under `output/`.

## Knowledge gaps and follow-up reading

- Papers under `output/papers/` should be reviewed for specific metric definitions and evaluation protocols:
  - `output/papers/Nair et al. - 2022 - DynaBARN Benchmarking Metric Ground Navigation in Dynamic Environments.pdf`
  - `output/papers/Biswas et al. - 2022 - SocNavBench A Grounded Simulation Testing Framework for Evaluating Social Navigation.pdf`
  - `output/papers/Francis et al. - 2023 - Principles and Guidelines for Evaluating Social Robot Navigation Algorithms.pdf`
  - `output/papers/Gouguet et al. - 2024 - Benchmarking Off-the-Shelf Human-Aware Robot Navigation Solutions.pdf`
  - Additional related papers in `output/papers/` (e.g., SCAND, Habitat 3.0, SocialNav-SUB) may inform scenario design and reporting.
- The benchmark should cross-check which metrics are emphasized in those papers and align with v1 requirements.

## Appendix: key local references

- Benchmark code: `robot_sf/benchmark/`
- Baselines: `robot_sf/baselines/`
- SocNav adapters: `robot_sf/planner/socnav.py`
- Scenario packs: `configs/scenarios/classic_interactions.yaml`, `configs/scenarios/francis2023.yaml`
- Maps: `maps/svg_maps/`, `maps/svg_maps/francis2023/`
- SocNavBench clone: `output/SocNavBench/`
- Arena-Rosnav clone: `output/arena-rosnav/`
