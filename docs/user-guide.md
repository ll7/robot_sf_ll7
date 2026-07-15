# Robot SF — User Guide

Welcome to Robot SF, a Gymnasium-based social-navigation simulation and benchmarking framework
for a robot moving through pedestrian-filled environments. This guide is the **task-oriented**
entry point for anyone who wants to *install, run, and use* Robot SF without diving into
internals or research methodology.

> New to the terminology? Start with the [Glossary](../glossary.md) — canonical definitions for
> acronyms and project terms (VRU, AMV, AMMV, SNQI, occluder, the evidence ladder, and more).

## 1. Install and first run

- [Quickstart Map](./quickstart.md) — first local checks, install, and one-command demos.
- [Development Guide](./dev_guide.md#setup) — full `uv sync --all-extras` setup, virtualenv, and
  pre-commit hooks.
- [Runtime Requirements](./dev_runtime_requirements.md) — non-`uv` host tools, Docker, GPU/SLURM,
  and the local capability checker.
- [Environment Configuration](../ENVIRONMENT.md) — detailed host environment setup and usage.

## 2. Run a demo

- [Examples Index](../examples/README.md) — quickstart, advanced, benchmark, and plotting workflows.
  + `examples/quickstart/01_basic_robot.py` — minimal robot simulation.
  + `examples/quickstart/02_trained_model.py` — run a trained model.
  + `examples/quickstart/03_custom_map.py` — load a custom SVG map.
- [Simulation View](./SIM_VIEW.md) — visualization and rendering system.

## 3. Load a map and create scenarios

- [SVG Map Editor](./SVG_MAP_EDITOR.md) — SVG-based map creation tools and usage.
- [OSM Map Generation](./osm_map_workflow.md) — reproducible maps from OpenStreetMap data.
- [Single Pedestrians](./single_pedestrians.md) — define individual pedestrians with goals or
  trajectories.
- [Map Verification](../specs/001-map-verification/quickstart.md) — validate SVG maps for structural
  integrity and runtime compatibility.
- [SVG Inspection Workflow](./dev/svg_inspection_workflow.md) — check route/zone consistency with
  `scripts/validation/svg_inspect.py`.
- [Scenario Zoo](./scenario_zoo/index.md) — maintained and emerging scenario families.
- [Scenario Specification Checklist](./scenario_spec_checklist.md) — authoring checklist for
  per-scenario/manifest files.

## 4. Choose and run a planner

- [Planner Zoo](./planner_zoo/index.md) — runnable, diagnostic-only, learned-policy, and blocked
  planner rows.
- [Planner Contribution Guide](./contributing_planner.md) — minimum path to add a planner.
- [Planner selection](./dev_guide.md#planner-selection-visibility-vs-classic-grid) — visibility vs
  classic grid global planners.
- [Global Planner quickstart](../specs/342-svg-global-planner/quickstart.md) — visibility-graph
  planner API, POI routing, integration guidance.

## 5. Run a benchmark

- [Social Navigation Benchmark Quickstart](../specs/120-social-navigation-benchmark-plan/quickstart.md)
  — step-by-step experiment execution, visualization, and interpretation.
- [Benchmark Runner And Metrics](./benchmark.md) — episode schema, aggregation, metrics (collisions,
  comfort exposure, SNQI), and the local smoke benchmark demo.
- [Local smoke benchmark demo](./benchmark.md): `uv run python scripts/demo/run_robot_sf_smoke.py`.
- [Mechanism-aware diagnostic reproduction](./benchmark.md):
  `uv run python scripts/demo/reproduce_mechanism_report.py --case topology-primary-route`.

## 6. Visualize results

- [Trajectory Visualization](./trajectory_visualization.md) — generate trajectory plots.
- [Force Field Visualization](./force_field_visualization.md) — heatmap + quiver figures (PNG/PDF).
- [Pareto Plotting](./pareto_plotting.md) — generate Pareto frontier plots.
- [Planner Tradeoff Plotting](./planner_tradeoff_plotting.md) — success/collision tradeoff figures.
- [Benchmark Visual Artifacts](./benchmark_visuals.md) — SimulationView & synthetic video pipeline.
- [SNQI Figures (orchestrator usage)](../examples/README.md) — SNQI-augmented figures from episodes.

## 7. Set up external datasets

- [External Data Setup Assistant](./external_data_setup.md) — license-safe staging and provenance
  manifests (Stanford Drone Dataset, SocNavBench, ETH/UCY, AMV calibration sources).
- [ETH/UCY External Trajectory Data](./datasets/eth-ucy.md) — acquisition, citation, expected layout.
- [SocNav Asset Setup](./socnav_assets_setup.md) — license-safe SocNav third-party dataset staging.
- [Real-World Trajectory Import](./real_world_trajectory_import.md) — Stanford Drone Dataset importer.

## 8. Troubleshoot

- [Debug Visualization](./debug_visualization.md) — visual debugging aids.
- [Telemetry Pane Display Fix](./telemetry-pane-fix.md) — continuous graph rendering and buffer
  management.
- [Runtime Requirements Checker](./dev_runtime_requirements.md) — inventory missing host tools.
- [Security Triage Guidance](./security_triage.md) — vulnerability reporting and dependency scanning.

## Where to go next

- **Want to publish a trustworthy result?** Move to the [Research & Benchmark Guide](./research-guide.md).
- **Want to extend the codebase (planners, scenarios, sim internals)?** Move to the
  [Developer Guide](./developer-guide.md).
