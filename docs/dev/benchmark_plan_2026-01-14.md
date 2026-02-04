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
- `robot_sf/planner/socnav.py` includes lightweight SocNav-style adapters (Sampling, Social Force, ORCA-like, SA-CADRL-like) and a fallback adapter that uses upstream SocNavBench if `third_party/socnavbench` is available.
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

- `third_party/socnavbench` includes a vendored subset of the upstream repository (planner + dependencies). The upstream sampling planner can be used via `SocNavBenchSamplingAdapter`.
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

#### Decision on benchmark scope and definition

We are benchmarking local planner with fixed global routes. The global planner is (both visibility graph and grid based) are sub-feature of a planned extension of adversarial testing.

We will try to use a merged set of scenarios. More scenarios mean a broader test of the capabilities. However, we currently don't know if all scenarios are completely suitable for our benchmark.

### B) Pipeline mismatch: synthetic runner vs. map-based Gym environment

Evidence:

- `robot_sf/benchmark/runner.py` uses `generate_scenario` (synthetic) and does not load SVG maps.
- `robot_sf/benchmark/full_classic/orchestrator.py` uses map-based Gym env but only a simple goal policy.

Why it matters:

- Local planner policies typically need a full environment (obstacles, routes, occupancy grids). The current benchmarking runner does not exercise those inputs, and the map-based benchmark is not yet wired to baseline planners.

Missing work:

- Decide on a single canonical runner and integrate baseline policies into it.
- Ensure the runner exposes observation formats suitable for local planners (structured SocNav + occupancy grid).

#### Decision on the environment pipeline

We want to use a runner that uses map-based gym environments with a combination of yaml scenario descriptions similar to `scripts/tools/render_scenario_videos.py`. Overall this file is already very good by using scenario descriptions. However, we are currently not evaluating any metrics and we should also be able to compare different local planner or RL policies.

### C) Scenario schema and validation do not cover map-based scenario YAMLs

Evidence:

- `robot_sf/benchmark/schema/scenarios.schema.json` only validates the synthetic scenario format.
- `configs/scenarios/classic_interactions.yaml` and `configs/scenarios/francis2023.yaml` use a different schema (map_file, simulation_config, single_pedestrians, metadata).

Why it matters:

- Without schema validation for the map-based scenarios, we cannot guarantee repeatability or catch schema drift.

Missing work:

- Define a canonical scenario schema that covers map-based scenario fields, or version two schemas with explicit selection rules.

#### Decision on the scenario schema

In my opinion the `configs/scenarios/francis2023.yaml` has superior functionality to define scenarios. We should mostly use this schema, but in the end we should converge to one schema that supports the functionality of both approaches.

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

#### Decision on the baseline algorithms

Ideally we should track all approaches and its current implementation in a single file. As a first approach, I would love to see all four algorithms as baseline:

- ORCA
- RVO
- TEB
- DWA

### E) Observation/action interface for local planners is not standardized

Evidence:

- SocNav structured observations exist, but benchmark runner uses synthetic numeric policies without standard observation objects.
- PPO adapter uses custom vector or image observation conversion; risks mismatch with local planner inputs.

Why it matters:

- If observation/action interfaces differ across planners, comparisons will be confounded. DynaBARN/BARN explicitly standardize robot size, sensors, and evaluation rules to avoid this.

Missing work:

- Define a single local-planner observation contract (e.g., structured SocNav + occupancy grid) and adapt all baselines to it.
- Define fixed action limits (v_max, omega_max) and ensure these are applied consistently.

#### Decision on observation/action interface (2026-01-14)

Decision:
- Use SocNav structured observation + occupancy grid (Option 2).
- Obstacle context is required for all benchmark scenarios.
- Local planners are evaluated only on waypoint following (global route fixed elsewhere).
- Provide up to 10 closest pedestrians, ordered by distance (closest-first), including velocities.

Options tracked:
- Option 1: SocNav structured only (no occupancy grid).
- Option 2: SocNav structured + occupancy grid (selected).
- Option 3: LiDAR/image-based observations (deferred).

Contract details to finalize:
Resolved (2026-01-14):
- Action space: unicycle only (v, omega).
- Observation frame: ego frame for occupancy grid and pedestrian velocities.
- Occupancy grid: 0.5 m resolution, 32x32 m extent, ego frame.
- Pedestrian ordering: distance ordering (closest-first) to prioritize near-field interaction risks.

Rationale — Action space (unicycle):
Unicycle commands align with the existing robot model and keep all planners comparable under the same kinematic limits. This avoids unfair advantages from holonomic controllers and keeps the evaluation grounded in realistic differential-drive constraints.

Rationale — Observation frame (ego frame):
Ego-frame inputs make the observation invariant to global map placement and reduce the amount of planner-specific coordinate handling. Aligning pedestrian velocities with the ego frame ensures consistency with occupancy grid semantics and simplifies policy inputs.

Rationale — Occupancy grid resolution/extent:
A 0.5 m resolution over 32x32 m balances local obstacle context with runtime cost and memory usage. It is large enough to capture near-term maneuvering constraints while staying small enough for real-time planner inference and stable baseline comparisons.

Rationale — Pedestrian ordering:
Ordering by distance (closest-first) emphasizes near-field interaction risk while remaining deterministic. This keeps the interface stable across runs without implying persistent pedestrian IDs or tracking.

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

#### Decision on metrics inputs (recommended for v1)

- Obstacles: populate `EpisodeData.obstacles` as a dense point cloud sampled from map geometry (fixed spacing, e.g., 0.25–0.5 m). This unblocks wall/clearing metrics without changing metric formulas; upgrade to segment-distance metrics later if needed.
- Shortest path: define shortest path length using the classic Theta* (v2) grid planner over the map geometry used by the environment, computed once per episode start/goal. If no path exists, return NaN and treat `path_efficiency` as NaN in aggregation.
- Force-gradient: treat `force_gradient_norm_mean` as optional for v1. Report NaN unless force-field grid sampling is explicitly enabled, and exclude it from any "core" summary table until wiring is complete.

### G) Reproducibility and provenance are not unified

Evidence:

- Two different episode schemas exist; the canonical schema does not include some fields the runner emits.
- PPO baseline may silently fall back to a goal-seeking policy when the model file is missing.

Why it matters:

- Reproducibility is a core promise of the benchmark (see repo constitution). Missing or ambiguous metadata reduces trust in results.

Missing work:

- Decide which schema is canonical and update all outputs to match it.
- Record algorithm loading status (e.g., "PPO fallback used") in `algorithm_metadata`.

#### Decision on reproducibility and provenance (recommended for v1)

- Canonical schema: `robot_sf/benchmark/schemas/episode.schema.v1.json` remains the single source of truth. Expand it to cover fields already emitted by the runner (`scenario_params`, `algorithm_metadata`, `config_hash`, `git_hash`, `timestamps`, `video`, `notes`) and ensure every episode record includes `version: "v1"`. Retire or mark `docs/dev/issues/social-navigation-benchmark/episode_schema.json` as historical to prevent drift.
- Algorithm metadata: require explicit load/fallback status in `algorithm_metadata` (e.g., `status: "ok" | "fallback" | "error"` plus `fallback_reason` when applicable). For PPO, record whether the model loaded and why fallback-to-goal was used.

### H) Scenario behavior semantics are incomplete

Evidence:

- Francis 2023 maps note missing speed tuning and behavior logic for overtaking, joining, waiting, etc.
- Map-based crowd behavior relies on ped density and routes, but not all scenario behaviors are implemented in the simulator.

Why it matters:

- Without behavior semantics, the scenarios may not reflect the intended interaction archetypes and comparisons across planners may be invalid.

Missing work:

- Implement the behavior hooks for single-ped scenarios (speed, wait, join/leave).
- Validate that each scenario achieves its intended interaction pattern.

#### Decision on scenario behavior semantics (recommended for v1)

- Scope: only behaviors that map to existing simulator controls are required for v1 (speed profiles, wait timers, join/leave triggers). Any scenario requiring unimplemented behaviors (e.g., multi-agent overtaking choreography) is tagged "unsupported" and excluded from the v1 benchmark set.
- Contract: each scenario YAML must declare a minimal behavior spec (speed profile + optional wait/join/leave) so the simulator can execute deterministically; missing fields fall back to documented defaults.
- Validation: add a lightweight scenario validation pass that checks expected behavior triggers (e.g., wait event fired, join/leave happened) and marks scenarios invalid if the pattern does not occur within the episode horizon.

### I) Evaluation protocol is not yet codified

Evidence:

- There are metrics and aggregation utilities, but no finalized benchmark protocol (seed counts, CI targets, acceptance criteria).
- The full classic benchmark includes adaptive precision thresholds but is not yet tied to local planner comparisons.

Why it matters:

- To compare local planners, we need a shared evaluation protocol (seeds, target CIs, and pass/fail criteria) similar to what BARN and DynaBARN provide.

Missing work:

- Define seed counts, CI thresholds, and minimal scenario coverage for benchmark runs.
- Decide whether adaptive sampling is required for all runs or only for full evaluation.

#### Decision on evaluation protocol (recommended for v1)

- Seeds: fixed seed count per scenario (e.g., 10) with deterministic seed list committed in configs to ensure reproducibility and comparability.
- Aggregation: bootstrap CIs at 95% with a documented sample count; report mean/median/p95 for all core metrics.
- Coverage: require all v1 scenario suites (classic interactions + map-based v1 set) unless explicitly marked unsupported; publish coverage table in the report.
- Adaptive sampling: disable by default for v1 comparisons; allow only in the full evaluation pipeline (flagged) so baseline runs remain deterministic and comparable.

### V1 scenario set (draft)

Goal: make the exact scenario list explicit so benchmark results are comparable.

- `configs/scenarios/classic_interactions.yaml` (all scenarios).
- `configs/scenarios/francis2023.yaml` (only scenarios whose behavior specs map to implemented hooks; unsupported scenarios are explicitly excluded and tagged).

Exclusion rules (v1):
- Scenarios requiring unimplemented behaviors (overtaking choreography, group splitting/merging, etc.) are marked unsupported.
- Scenarios that fail the behavior-validation pass within the episode horizon are excluded until fixed.

### Episode schema migration plan (v1)

- Expand `robot_sf/benchmark/schemas/episode.schema.v1.json` to include fields emitted by the runner (`scenario_params`, `algorithm_metadata`, `config_hash`, `git_hash`, `timestamps`, `video`, `notes`).
- Ensure all episode records include `version: "v1"` and are validated on write.
- Provide a one-off migration helper (or documented command) to upgrade existing JSONL files by injecting the missing keys with explicit defaults.
- Mark `docs/dev/issues/social-navigation-benchmark/episode_schema.json` as historical to prevent drift.

### Core metrics for v1 (reporting table)

Core (always reported when inputs exist):
- success, time_to_goal_norm, collisions, near_misses, min_distance, mean_distance, path_efficiency
- avg_speed, jerk_mean, curvature_mean, energy
- force_q50/q90/q95, force_exceed_events, comfort_exposure

Conditional (reported only when inputs are available):
- wall_collisions, clearing_distance_min, clearing_distance_avg

Optional (v1 excluded unless explicitly enabled):
- force_gradient_norm_mean

### Seed list location (v1)

- Commit a deterministic seed list per scenario set under `configs/benchmarks/seed_list_v1.yaml` (or equivalent).
- The runner reads the list directly; no ad-hoc seed generation in scripts.

### Baseline inclusion bar (v1)

To be included in the v1 benchmark, a baseline must:
- Run headless and deterministically with the SocNav+occupancy observation contract.
- Expose config + metadata (hashable) and report fallback status when applicable.
- Complete the scenario suite within the evaluation time budget.
- Avoid external dependencies that are unavailable or unlicensed for redistribution.

### Decision record (keep current)

| Date       | Decision area | Decision summary | Owner | Doc link |
|------------|----------------|------------------|-------|----------|
| 2026-01-14 | Metrics inputs | Obstacle sampling, Theta* shortest path, force-gradient optional | TBD | this doc |
| 2026-01-14 | Reproducibility | Canonical schema expanded; explicit fallback status | TBD | this doc |
| 2026-01-14 | Behavior semantics | Limited behavior scope + validation pass | TBD | this doc |
| 2026-01-14 | Evaluation protocol | Fixed seeds + bootstrap CIs + coverage table | TBD | this doc |

## Open questions (decisions needed before implementation)

Resolved (2026-01-14 decisions):
1) Pipeline authority: map-based Gym runner with YAML scenarios (merged set).
2) Local policy definition: local planner only, fixed global routes.
3) Observation/action contract: SocNav structured + occupancy grid, unicycle actions.
4) Metrics optionality: force-gradient optional; obstacle/path metrics required once inputs exist.
5) Shortest-path efficiency: Theta* (v2) shortest path length.
6) Seed count and CI targets: fixed seed list per scenario (e.g., 10), 95% bootstrap CIs.

Still open:
1) Which scenario suites are included in v1 (exact YAML list + exclusion rules)?
2) Which baseline algorithms are required vs. optional for v1 (ORCA/RVO/TEB/DWA feasibility/inclusion bar)?
3) How should model artifacts (PPO, imitation) be versioned and validated?
4) Do we need compatibility layers to compare with external benchmarks (SocNavBench joystick API, DynaBARN/BARN formats)?

## Proposed structure for the benchmark effort (phased)

### Phase 0: Decision record (scope + schema)

- Choose canonical benchmark pipeline and scenario schema.
- Fix the baseline algorithm set for v1.0 and document inclusion/exclusion rationale.
- Finalize the local planner observation/action contract.

Acceptance criteria:
- Decision record updated with resolved items and remaining open questions.
- Canonical schema and scenario list are referenced by path and version.

Risks:
- Overlapping decisions across docs cause drift if not consolidated.

### Phase 1: Unify runner and scenario schema

- Implement a single runner that can load map-based scenarios and execute local planners.
- Add schema validation for map-based scenario YAMLs.

Acceptance criteria:
- Map-based scenarios validate against a canonical schema.
- Runner executes at least one scenario suite end-to-end with a baseline policy.

Risks:
- Schema migration breaks existing YAMLs or examples.

### Phase 2: Baseline integration and metadata

- Wire SocNav-style planners (Sampling, Social Force, ORCA-like, SA-CADRL-like) into the benchmark registry.
- Add at least one classical local planner baseline (DWA or TEB) if feasible.
- Record detailed `algorithm_metadata` (config hash, model path, fallback flags).

Acceptance criteria:
- Baseline registry includes at least one classical planner and one reciprocal-avoidance planner.
- Episode records include algorithm status + fallback reason when applicable.

Risks:
- External dependencies/licensing block planner integration.

### Phase 3: Metrics plumbing

- Populate obstacle data from SVG map geometry to enable wall collisions and clearing distance metrics.
- Compute map-aware shortest path for path efficiency.
- Decide if force-field gradient metrics are in or out for v1.

Acceptance criteria:
- Wall/clearing metrics are non-NaN for obstacle maps.
- Path efficiency uses map-aware shortest paths and is reproducible.

Risks:
- Geometry/coordinate mismatches produce incorrect distances or collisions.

### Phase 4: Protocol and reporting

- Define the evaluation protocol: seeds, CI thresholds, success criteria.
- Produce baseline stats for SNQI normalization.
- Lock the minimal scenario suite and publish benchmark artifacts under `output/`.

Acceptance criteria:
- Published report includes coverage table, seed list, and CI settings.
- SNQI baseline stats are versioned and referenced by the report.

Risks:
- Protocol changes after baselines run invalidate prior results.

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
- SocNavBench vendored subset: `third_party/socnavbench/`
- Arena-Rosnav clone: `output/arena-rosnav/`
