# Robot SF — Developer Guide

This guide is the **developer** entry point for Robot SF. It covers architecture, contribution
workflow, validation gates, CI, and internals. It assumes you can already install and run Robot SF
(see the [User Guide](./user-guide.md)); it focuses on *building and extending* the codebase rather
than running experiments (see the [Research & Benchmark Guide](./research-guide.md) for that).

## 1. Setup and local workflow

- [Development Guide](./dev_guide.md) — central contributor reference: setup, testing, quality gates,
  coding standards.
- [Runtime Requirements](./dev_runtime_requirements.md) — non-`uv` host tools and the capability
  checker.
- [Environment Configuration](./ENVIRONMENT.md) — detailed host environment setup.
- [UV Migration Notes](./UV_MIGRATION.md) — migration to the `uv` package manager.
- [Subtree Migration Guide](./SUBTREE_MIGRATION.md) — git subtree integration for fast-pysf.
- [Contributing](./../CONTRIBUTING.md) — contributor workflow and conventions.
- [Agents & Contributor Onboarding](../AGENTS.md) — repository structure, coding/testing conventions,
  workflow tips.
- [Agent Workflow Entrypoints](./ai/agent_workflow_entrypoints.md) — correct `uv run` patterns and
  validation entrypoints.
- [Agent Run Manifest](./agent_run_manifest.md) — making agent-assisted runs auditable.

## 2. Architecture and internals

- [Architecture Decision Records](./adr) — ADR index.
- [Configuration Architecture](./architecture/configuration.md) — configuration hierarchy and
  precedence.
- [Refactoring Overview](./refactoring/) — refactored environment architecture (status, plan,
  migration guide, summary).
- [Repository Structure Analysis](./dev/issues/repository-structure-analysis.md) — codebase
  organization assessment and roadmap.
- [Architectural Decoupling (Feature 149)](../specs/149-architectural-coupling-and/) — simulator
  facade and registries.
- [Environment Factory Migration (Feature 130)](./dev/issues/130-improve-environment-factory/migration.md)
  — factory-pattern migration guidance.
- [Benchmark Design](./dev/issues/social-navigation-benchmark/README.md) — benchmark design, metrics,
  schema, run commands.
- [Occupancy Grid Guide](./dev/occupancy/Update_or_extend_occupancy.md) — configure grid observations.
- [Helper Catalog](./dev/helper_catalog.md) — reusable render helpers.

## 3. Validation gates and CI

- [Coverage Guide](./coverage_guide.md) — coverage collection, baseline tracking, CI integration.
- [Per-Test Performance Budget](./dev_guide.md#per-test-performance-budget) — soft/hard test timeouts.
- [Code Review Guide](./code_review.md) — benchmark-facing review criteria and regression traps.
- [Performance Notes](./performance_notes.md) — performance targets and optimization notes.
- [Benchmark Planner Quality Audit](./benchmark_planner_quality_audit.md) — planner decision table.
- [Benchmark/Planner Review Guide](./code_review.md) — semantics, normalization, provenance review.
- [Mutation Testing Triage](../mutation_testing_triage.md) — mutation-testing triage notes.

## 4. Planner and scenario contribution

- [Planner Contribution Guide](./contributing_planner.md) — minimum planner/adapter contribution
  flow, registry, and benchmark boundary.
- [Planner Adapter Starter Template](./dev/planner_adapter_template.md) — copy-and-adapt reference.
- [Scenario Specification Checklist](./scenario_spec_checklist.md) — per-scenario authoring checklist.
- [Scenario Zoo](./scenario_zoo/index.md) — scenario family index.
- [Policy Search Context](./context/policy_search/README.md) — file-based local policy-search workflow.

## 5. Tooling and developer scripts

- [SNQI Weight Tools](./snqi-weight-tools/README.md) — recompute, optimize, analyze SNQI weights.
- [Pyreverse UML Generation](./pyreverse.md) — class diagrams from code.
- [Data Analysis Utilities](./DATA_ANALYSIS.md) — analysis helpers and data processing tools.
- [Imitation Results Analysis](./imitation_results_analysis.md) — baseline vs pre-trained comparisons.
- [Imitation Learning Pipeline](./imitation_learning_pipeline.md) — training pipeline overview.
- [Dev scripts](./../scripts/dev/) — shared development entry points (ruff, tests, PR readiness).
- [SVG Inspection Workflow](./dev/svg_inspection_workflow.md) — route/zone consistency checks.

## 6. Visualization and rendering internals

- [Simulation View](./SIM_VIEW.md) — visualization and rendering system.
- [LiDAR Configuration Reference](./lidar_configuration.md) — canonical robot/ego scan defaults.
- [SVG Map Editor](./SVG_MAP_EDITOR.md) — SVG map creation tooling.
- [OSM Map Generation](./osm_map_workflow.md) — programmatic map generation from OpenStreetMap.
- [Figure Naming Scheme](./dev/issues/figures-naming/design.md) — canonical figure folder naming.

## 7. Pedestrian metrics internals

- [Pedestrian Metrics Overview](./ped_metrics/PED_METRICS.md) — implemented metrics and purpose.
- [Metric Analysis](./ped_metrics/PED_METRICS_ANALYSIS.md) — metrics used in research and validation.
- [NPC Pedestrian Design](./ped_metrics/NPC_PEDESTRIAN.md) — NPC pedestrian design and behavior.
- [Pedestrian Density Reference](./ped_metrics/PEDESTRIAN_DENSITY.md) — density canonical triad.

## Where to go next

- **Need to run or use Robot SF?** See the [User Guide](./user-guide.md).
- **Need benchmark semantics or evidence grading?** See the [Research & Benchmark Guide](./research-guide.md).
