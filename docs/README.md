# Robot SF Documentation

Welcome to the Robot SF documentation! This directory contains comprehensive guides and references for using and developing with the Robot SF simulation framework.

**New to the terminology?** Start with the [Glossary](./glossary.md) — the canonical definitions for acronyms and project-specific terms (VRU, AMV, AMMV, SNQI, occluder, the evidence ladder, and more).

<!-- This document should mainly serve as a navigation hub and overview for the various components and guides available within the Robot SF project. Refer to individual files for detailed information. -->

**Artifact root**: All generated artifacts (JSONL, figures, videos) must live under the git-ignored `output/` directory. Legacy `results/` paths have been migrated; update commands accordingly when running examples or scripts. Small, reviewable copies of durable evidence may be promoted into [docs/context/evidence](./context/evidence/README.md); do not mirror `output/` wholesale.

**Route-clearance caveats**: paper-facing preflights expose route-clearance certifications from
[`configs/benchmarks/route_clearance_certifications_v1.yaml`](../configs/benchmarks/route_clearance_certifications_v1.yaml);
see [issue_1105_route_clearance_certification.md](./context/issue_1105_route_clearance_certification.md).

## Static Docs Site

This repository includes a lightweight Sphinx site over the existing Markdown docs. Build it from
the repository root with:

```bash
uv run --group docs sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` for the browsable navigation layer. The site is intentionally
thin: existing Markdown files remain the source of truth, and generated HTML under
`docs/_build/` is disposable local output.

## 🚀 Social Navigation Benchmark Platform (Complete)

**The Social Navigation Benchmark Platform is now fully operational!**

### Quick Start

* **[Complete Quickstart Guide](../specs/120-social-navigation-benchmark-plan/quickstart.md)** - Step-by-step experiment execution, visualization, and interpretation
* **[CLI Reference](./dev/issues/social-navigation-benchmark/README.md)** - All 15 CLI subcommands with examples
* **Implementation Status**: All major features complete, 108 tests passing

### Core Capabilities

* **Episode Runner**: Parallel execution with resume functionality and deterministic seeding
* **Metrics Suite**: SNQI composite index with component breakdown and weight recomputation
* **Baseline Interface**: Unified PlannerProtocol for SocialForce, PPO, Random planners
* **Statistical Analysis**: Bootstrap confidence intervals and robust aggregation
* **Figure Orchestrator**: Distribution plots, Pareto frontiers, force fields, thumbnails, tables
* **CLI Tools**: 30 subcommands covering full experiment workflow

### Ready-to-Use Workflows

1. **Quick Assessment** (~15 min): Compare robot policies against baselines
2. **Research Study** (~2-4 hours): Multi-parameter analysis with publication figures
3. **Weight Sensitivity** (~45 min): Analyze SNQI component importance

**Start Here**: [specs/120-social-navigation-benchmark-plan/quickstart.md](../specs/120-social-navigation-benchmark-plan/quickstart.md)

---

- [🚀 Social Navigation Benchmark Platform (Complete)](#-social-navigation-benchmark-platform-complete)
  - [Quick Start](#quick-start)
  - [Core Capabilities](#core-capabilities)
  - [Ready-to-Use Workflows](#ready-to-use-workflows)
- [📚 Documentation Index](#-documentation-index)
  - [Getting Started](#getting-started)
  - [Architecture Decision Records](#architecture-decision-records)
  - [Benchmarking \& Metrics](#benchmarking--metrics)
  - [Tooling](#tooling)
  - [Architecture \& Refactoring](#architecture--refactoring)
  - [Simulation \& UI](#simulation--ui)
  - [Figures \& Visualization](#figures--visualization)
  - [Performance \& CI](#performance--ci)
  - [Hardware \& Environment](#hardware--environment)
  - [Additional Resources (Legacy Structure)](#additional-resources-legacy-structure)
  - [🏗️ Architecture \& Development](#️-architecture--development)
  - [🎮 Simulation \& Environment](#-simulation--environment)
  - [📊 Analysis \& Tools](#-analysis--tools)
    - [Social Navigation Benchmark (Overview)](#social-navigation-benchmark-overview)
    - [Figures naming and outputs](#figures-naming-and-outputs)
    - [LaTeX Table Embedding (SNQI / Benchmark Tables)](#latex-table-embedding-snqi--benchmark-tables)
  - [Per-Test Performance Budget](#per-test-performance-budget)
  - [⚙️ Setup \& Configuration](#️-setup--configuration)
  - [📈 Pedestrian Metrics](#-pedestrian-metrics)
  - [📁 Media Resources](#-media-resources)
- [🚀 Quick Start Guides](#-quick-start-guides)
  - [New Environment Architecture (Recommended)](#new-environment-architecture-recommended)
  - [Legacy Pattern (Still Supported)](#legacy-pattern-still-supported)
    - [Environment Factory Ergonomics Migration (Feature 130)](#environment-factory-ergonomics-migration-feature-130)
- [🎯 Key Features](#-key-features)
  - [Environment System](#environment-system)
  - [Simulation Capabilities](#simulation-capabilities)
  - [Training \& Analysis](#training--analysis)
- [📖 Documentation Highlights](#-documentation-highlights)
  - [🆕 Latest Updates](#-latest-updates)
  - [📋 Migration Status](#-migration-status)
  - [Architecture \& design features](#architecture--design-features)
- [🔗 External Links](#-external-links)
- [🤝 Contributing](#-contributing)
- [📞 Support](#-support)
- [Planner Documentation](#planner-documentation)

## 📚 Documentation Index

### Getting Started

* **[Development Guide](./dev_guide.md)** - Primary reference for development workflows, setup, testing, quality gates, and coding standards
* **[Maintainer Values And Hard Contracts](./maintainer_values.md)** - Compact source of truth for current values: honest, transparent, reproducible progress; exploration labels; uncertainty and validation policy
* **[Runtime Requirements](./dev_runtime_requirements.md)** - Non-`uv` host tools, system packages, optional Docker/`gh-act` support, and the local capability checker
* **[Security Triage Guidance](./security_triage.md)** - Vulnerability reporting, dependency scanning, static-analysis triage, and accepted-risk handling for research code
* **[External Data Setup Assistant](./external_data_setup.md)** - License-safe local staging and compact provenance manifests for external datasets including Stanford Drone Dataset, SocNavBench, ETH/UCY, and AMV calibration-source assets
* **[ETH/UCY External Trajectory Data](./datasets/eth-ucy.md)** - Public acquisition, citation, and expected layout notes for locally staged ETH BIWI and UCY Crowds-by-Example trajectories
* **[Context Retrieval Index](./context/INDEX.md)** - Retrieval-first catalog for current context-note entry points, status rules, optional context tools, and curated context-pack scopes
* **[Agent Workflow Entrypoints And Large-File Navigation](./ai/agent_workflow_entrypoints.md)** - Correct `uv run` command patterns, validation entrypoints, model registry path, and targeted large-file reading guidance for agents
* **[Agent Run Manifest](./agent_run_manifest.md)** - Lightweight `agent_run_manifest.yaml` convention for making substantial agent-assisted runs auditable: when it is required, where to store it, trace/log hygiene, and a copyable template
* **[Issue #2013 Backend Adapter Contract](./context/issue_2013_backend_adapter_contract.md)** - Required adapter fields, fail-closed behavior, and claim boundaries for alternate simulator backend integration
* **[Context Notes Workflow](./context/README.md)** - Canonical rules for linked Markdown handoff notes, note updates vs new notes, stale-note handling, and discoverability
* **[Planner Contribution Guide](./contributing_planner.md)** - Minimum path for adding a planner with adapter/protocol metadata, config-first invocation, smoke proof, registry status, and benchmark boundaries
* **[Planner Zoo](./planner_zoo/index.md)** - User-facing index of runnable, diagnostic-only, learned-policy, monitor-only, and blocked planner rows from the policy-search registries
* **[Open-Issues Implementation Status](./context/open_issues_implementation_status_2026-05-12.md)** - Handoff record for the May 2026 open-issues pass, including implemented slices, blocked items, and remaining follow-up surface
* **[Open-Issues Maintainer Input Triage](./context/open_issues_maintainer_input_triage.md)** - Consolidated maintainer-decision inventory for open issues that still need scope, contract, or prioritization guidance
* **[Open-Issues PR Split Strategy](./context/open_issues_pr_split_strategy_2026-05-13.md)** - PR packaging strategy and validation grouping for the open-issues implementation pass
* **[Project Prioritization](./project_prioritization.md)** - Priority-score model, Project #5 field semantics, and the local/manual score-sync workflow
* **[GitHub Workflow Batching](./context/issue_713_batch_first_issue_workflow.md)** - Batch issue cleanup first, defer Project #5 routing, and run derived score sync last
* **[Goal-Driven Agent Loops](./context/goal_driven_agent_loops_2026-05-13.md)** - Shared contract for autonomous issue discovery, issue implementation, PR review, and user-in-the-loop issue audit skills
* **[Artifact Evidence Vocabulary](./context/artifact_evidence_vocabulary.md)** - Shared issue/PR vocabulary for exploratory outputs, durable evidence, release artifacts, benchmark claims, and paper-facing claims
* **[Question-First Experiment Registry](../experiments/README.md)** - Register experiment intent, canonical configs, expected artifacts, and validation gates before launching runs
* **[Policy Search Portfolio Overview](./context/policy_search/portfolio_overview_2026-05-05.md)** - Current non-training policy-search portfolio ranking, promotion status, and h500 horizon evidence pointers
* **[Agent Index](./AGENT_INDEX.md)** - Agent-oriented index of training, benchmarking, observations, and artifacts
* **[AI Repo Overview](./ai/repo_overview.md)** - Short orientation for Codex-style agents: where to read first, core repo areas, and common failure modes
* **[Understand-Anything Knowledge Graph](./ai/understand_anything.md)** - Shared graph artifact,
  Codex setup, dashboard usage, update flow, and Git LFS policy
* **[Issue #1181/#1191 `ml-intern` Workflow Extraction](./context/issue_1181_ml_intern_experiment_assistant.md)** -
  Safe-use boundary and 2026-05-15 decision to extract `ml-intern` workflow ideas into Robot SF's
  Codex-native proof-first practice instead of running the CLI smoke by default
* **[Issue #1179 CARLA Docker Runtime](./context/issue_1179_carla_docker_runtime.md)** - Pinned `carlasim/carla:0.9.16` preflight/smoke command, current local Docker smoke status, and boundary before live replay semantics
* **[Issue #1111 CARLA Setup-Only Smoke](./context/issue_1111_carla_setup_smoke.md)** - Ephemeral `carla==0.9.16` setup-only T1 smoke proof and boundary before Issue #1169 live replay
* **[Issue #1239 Human-Model Transfer Robustness](./context/issue_1239_human_model_transfer.md)** - Explicit human-model variant/source metadata and a conservative transfer-smoke benchmark config
* **[Issue #1169 CARLA Live T1 Oracle Replay](./context/issue_1169_carla_live_replay.md)** - Docker-backed live replay command, real CARLA `0.9.16` client/server connection, and fail-closed static-geometry boundary
* **[Issue #1344 Paired AMV Primary Protocol](./context/issue_1344_paired_amv_protocol_report.md)** - Rerunnable paired nominal/stress AMV primary-row protocol, compact evidence, and interpretation boundary before all-runnable expansion or paper-facing claims
* **[Issue #2001 AMV Actuation Proxy Source Analysis (2026-06-01)](./context/issue_2001_amv_actuation_proxy_source_analysis.md)** - Platform-class proxy-source decision for AMV actuation provenance, accepting public e-scooter longitudinal acceleration/braking evidence while preserving missing yaw, angular-acceleration, and latency fields
* **[Issue #1398 Metric Rollup Reconciliation](./context/issue_1398_metric_rollup_reconciliation.md)** - Analyzer SNQI row-vs-episode reconciliation and claim-scope boundary for #1344/#1354 follow-up evidence
* **[Issue #1433 Adversarial Edge-Case Search Design (2026-05-22)](./context/issue_1433_adversarial_edge_case_search_design.md)** - Bounded v1 design for crossing/TTC adversarial search with parameter bounds, invalid-candidate handling, scripted vs learned decisions, execution contract, failure classes, artifact policy, and explicit dependency on Issue #1434 uncertainty/coverage reporting
* **[Issue #2468 Adversarial Scenario Generation Roadmap (2026-06-07)](./context/issue_2468_adversarial_generation_roadmap.md)** - Cross-method roadmap for bounded search, RL adversaries, diffusion/generative manifests, learned proposal models, and LLM-to-structured-manifest assistants, with common controls, validity gates, manifest smoke/quality prerequisites, and the #2568 learned-expansion gate
* **[Issue #1457 Adversarial Map And Start-State Generation Protocol (2026-05-23)](./context/issue_1457_adversarial_generation_protocol.md)** - Conservative route/start-state-first protocol for adversarial stress-test generation, with fail-closed case validity rules and a compact seeded smoke summary on `classic_head_on_corridor_low`
* **[Issue #1237 Adversarial Failure Archive](./context/issue_1237_adversarial_failure_archive.md)** - Compact `adversarial_failure_archive.v1` manifests for deterministic adversarial failure grouping and replay pointers without copying raw bundles
* **[Issue #4360 Adversarial Dispatchable Inventory](./context/issue_4360_adversarial_dispatchable_inventory.md)** - Current adversarial pedestrian hooks, repeatable seeds/configs, runner assumptions, and how-to-run boundary for the dispatchable half only
* **[Issue #1500 Adversarial Campaign Manifest Freeze](./context/issue_1500_adversarial_manifest.md)** - Frozen scenario/search/budget manifest contract for the bounded adversarial comparison campaign, with non-evidence row classification and tracked checksum evidence
* **[Issue #1432 Adaptive Test Strategy Claim Audit](./context/adaptive_test_claim_audit_2026-05.md)** - Docs-only claim hygiene audit: method inventory, Bayesian/Optuna tooling classification, and conservative verdict on "adaptive test strategy" versus "bounded adversarial/scenario exploration"
* **[AI Coding Workflow](./ai/ai-workflow.md)** - End-to-end AI issue-to-PR workflow, validation gates, review loop, and traceability conventions
* **[PR First-Pass Review Audit](./context/pr_first_pass_review_audit_2026-05-14.md)** - Recent merged-PR review findings and the pre-opening self-review checklist for reducing repeated reviewer fixes
* **[AI Planner Zoo Context](./ai/planner_zoo_context.md)** - Planner-zoo integration context, readiness framing, and provenance/adapter questions to answer explicitly
* **[AI Context Packing Decision](./ai/context_packing.md)** - Current decision and rationale for using Repomix as the default focused-context bundler
* **[Awesome Copilot Adaptation](./ai/awesome_copilot_adaptation.md)** - Selective adaptation plan for Codex skills, including `autoresearch`,   `auto-improvement`, context-discovery, quality, and doc-sync skills
* **[AI Retrieval Deferral Note](./ai/retrieval_deferral.md)** - Why MCP/retrieval layers stay out of scope until a real context boundary appears
* **[Agent Memory Index](../memory/MEMORY.md)** - Repo-local Markdown memory taxonomy for durable agent context, with linked architecture, decision, experiment, failure, and benchmark notes
* **[Issue #1151/#1219 Manual-Control MVP](./context/issue_1151_manual_control_mvp_foundation.md)** - Pygame manual-control benchmark recorder foundations plus the local `scripts/manual_control/run_pygame_session.py` runner, JSONL recording, session manifest, and headless smoke command
* **[Issue #1163 Manual-Control Recording Format Decision](./context/issue_1163_manual_control_recording_format.md)** - No-change decision for compact manual-control recording formats, with JSONL size/throughput thresholds and provenance requirements for any future derived artifact
* **[Issue #1154 Web-Game Data Collection Path](./context/issue_1154_web_game_data_collection_path.md)** - Deferred web-game data collection follow-up; rationale for keeping schema parity with the local recorder and sequenced implementation order
* **[Observation Contract](./dev/observation_contract.md)** - Observation schemas, shapes, and normalization conventions
* **[Holonomic Action Contract](./dev/holonomic_action_contract.md)** - Exact holonomic action-space semantics, heading behavior, and benchmark bridge rules
* **[Helper Catalog](./dev/helper_catalog.md)** - Reusable environment, policy, episode, rendering, and docs helpers extracted from examples and benchmark scripts
* **[Training Protocol Template](./dev/training_protocol_template.md)** - Fill-in template for documenting training/evaluation runs
* **[Canonical PPO Training Workflow](./training/ppo_training_workflow.md)** - Config-driven PPO entrypoint, evaluation cadence semantics, and startup provenance logging.
* **[Robot SF Environment Contract And Training Provenance](./training/environment_contract.md)** - Factory entrypoints, rollout ownership, reward-versus-benchmark boundary, and PPO run-record checklist.
* **[Vectorized Environment Support](./training/vectorized_env_support.md)** - Public DummyVecEnv/SubprocVecEnv support matrix, spawn start-method contract, and smoke-test boundary.
* **[Issue #1037 RL Environment Patterns](./context/issue_1037_rl_environment_patterns.md)** -
  Design note mapping May 2026 LLM-era RL environment patterns to Robot SF training, reward,
  rollout, benchmark, scaling, and provenance boundaries.
* **[SLURM Submission Workflow](./dev/slurm_submission.md)** - Submit batch jobs with the effective partition/QoS max wall time by default
* **[SLURM Multi-Worktree Branch Workflow](./context/slurm_multi_worktree_branch_workflow.md)** - Submit jobs from multiple active branches on one login node without branch-switch ambiguity
* **[SLURM Resource Audit](./dev/slurm_resource_audit.md)** - Inspect Slurm allocations, query W&B system metrics correctly, and decide whether CPU, memory, or GPU requests are oversized
* **[Model Registry](../model/registry.md)** - Track trained policies and load them on-demand via `robot_sf.models`
* **[Model Registry Publication Workflow](./model_registry_publication.md)** - Preserve promoted/paper-facing policies as public GitHub release assets with checksums and registry pointers
* **[Examples Catalog](../examples/README.md)** - Manifest-backed index of quickstart, advanced, benchmark, and plotting scripts with usage metadata
* **[SocNav structured observation example](../examples/advanced/18_socnav_structured_observation.py)** - Run RobotEnv with SocNavBench-style observations and a simple planner adapter.
* **[SocNav structured observation how-to](./dev/issues/socnav_structured_observation.md)** - Enable `ObservationMode.SOCNAV_STRUCT` and use planner adapters (lightweight + SocNavBench wrapper).
* **[Issue 403 Grid PPO Training Runbook](./training/issue_403_grid_training.md)** - Step-by-step training for the grid+SocNav PPO expert.
* **[PPO num_envs Benchmark (imech156-u)](./training/ppo_num_envs_benchmark_imech156u.md)** - Host utilization, throughput, and stability benchmark for PPO `num_envs` sizing on imech156-u.
* **[Predictive Planner Training Runbook](./training/predictive_planner_training.md)** - Data collection, training, proxy selection, and benchmark evaluation workflow for `prediction_planner`.
* **[Issue #1138 Predictive Obstacle Feature Schema](./context/issue_1138_predictive_obstacle_features_schema.md)** - Stable six-value obstacle-feature contract, sentinel behavior, and map-derived lifecycle follow-up for predictive planner inputs
* **[BR-07 Evening Run: Predictive Planner Refresh](./training/br07_predictive_evening_run.md)** - Reproducible evening-run checklist for predictive planner refresh, evaluation, and promotion artifacts.
* **[Issue 708 Main-Based PPO Retrain Campaign](./context/issue_708_main_based_ppo_retrain_campaign.md)** - Final no-promotion recommendation for the issue-708 PPO campaign family, plus the original retrain config, SLURM submission path, deterministic eval surface, and provenance record.
* **[Issue #749 BC-Preinitialized PPO Launch Packet](./context/issue_749_bc_preinit_ppo_launch_packet.md)** - Config-first BC warm-start PPO challenger path, artifact boundary, and follow-up execution gate for the v10 fine-tune contract
* **[Issue #1209 Imitation Observation Contract](./context/issue_1209_imitation_observation_contract.md)** - BR-06 checkpoint-compatible collection, BC, and PPO fine-tune observation-contract proof path for unblocking #1108
* **[Issue 791 Promotion Campaign](./context/issue_791_promotion_campaign_128k_256k.md)** - Medium- and long-horizon ablation campaign status, GPU predictive-foresight fix, active long runs, and follow-up boundaries.
* **[Issue 856 PPO All-Scenarios Full-Budget Campaign](./context/issue_856_ppo_all_scenarios_full_budget.md)** - Seed-123 broad-training submission, replica gate, and benchmark-comparison plan against the eval-aligned leader.
* **[Issue 863 SVG/Model Log Spam](./context/issue_863_svg_model_log_spam.md)** - Log dedupe and PPO evaluation phase-marker decision for issue-791 long-run triage.
* **[Issue 739 PPO Ablation Stage 1](./context/issue_739_ppo_ablation_stage1.md)** - Final reward/observation/optimizer ablation outcome: no tested simplification improved over the issue-708 baseline, with a pointer to the Stage 2 decision record.
* **[Predictive Planner Complete Tutorial](./training/predictive_planner_complete_tutorial.md)** - Full concept-to-code tutorial (model, scoring, risk-adaptive search, diagnostics, and reproducibility)
* **[DreamerV3 RLlib Runbook (`drive_state` + `rays`)](./training/dreamerv3_rllib_drive_state_rays.md)** - Config-first training flow for RLlib DreamerV3 without image observations.
* **[Issue #1190 DreamerV3 Checkpoint Import Boundary](./context/issue_1190_dreamerv3_checkpoint_import_boundary.md)** - Fail-closed BR-08 warm-start probe showing that Ray 2.53.0 has no clean Robot SF world-model import contract.
* **[Global Planner Quickstart (WIP)](../specs/342-svg-global-planner/quickstart.md)** - Placeholder for the upcoming SVG-based global planner documentation and examples.
* **[Artifact Policy Quickstart](../specs/243-clean-output-dirs/quickstart.md)** - Step-by-step migration, guard enforcement, and override instructions for the canonical `output/` tree
* **[Imitation Learning Pipeline](./imitation_learning_pipeline.md)** - Complete guide to PPO pre-training with expert trajectories
* **[Imitation Learning Quickstart](../specs/001-ppo-imitation-pretrain/quickstart.md)** - Step-by-step workflow for BC pre-training and PPO fine-tuning
* **[Optuna Expert PPO Sweep Report (2026-02-11)](./training/optuna_expert_ppo_sweep_2026-02-11.md)** - Sweep findings and PPO hyperparameter guidelines
* **[Waypoint Noise For Route Generalization](./training/waypoint_noise.md)** - Configure Gaussian waypoint perturbation to reduce route memorization during training
* **[Research Reporting](./research_reporting.md)** - Automated research report generation: multi-seed aggregation, statistical analysis, figure generation, Markdown/LaTeX export
* **[Feature Extractors Guide](./feature_extractors/usage_guide.md)** - Configure and compare extractor presets, run multi-extractor training, and generate reports
* **[Run Tracker & History CLI](./dev_guide.md#run-tracker--history-cli)** - Enable the failure-safe tracker on the imitation pipeline, monitor `status`/`watch` output, run telemetry perf-tests, mirror telemetry to TensorBoard, filter historical runs, and export Markdown/JSON summaries via the `scripts/tools/run_tracker_cli.py` commands (`status`, `watch`, `list`, `summary`, `export`, `perf-tests`, `enable-tensorboard`)
* **[Issue #845 Slurm Utilization Probe](./context/issue_845_slurm_utilization_probe.md)** - Collect `sstat`/`sacct`/`seff` evidence for low CPU-utilization investigations without launching new jobs

### Architecture Decision Records

* **[ADR Index](./adr/README.md)** - Lightweight process and index for durable architecture and long-lived contract/process decisions
* **[ADR Template](./adr/template.md)** - Copy-ready format for source-backed ADRs (status, context, alternatives, impacts)

### Benchmarking & Metrics

* **[Benchmark Spec (Classic Interactions)](./benchmark_spec.md)** - Scenario split + seeds, baseline categories, reproducible commands, and metric caveats
* **[Benchmark Scenario And Model Governance](./benchmark_governance.md)** - PR review contract for scenarios, metrics, model profiles, versioned schemas, release-bound evidence, and deprecated/superseded scenarios
* **[Assurance Fragments](./assurance_fragments.md)** - Machine-readable claim-argument-evidence fragments for campaign artifacts, including release-gate to GSN argument-node mapping
* **[Scenario Zoo Index](./scenario_zoo/index.md)** - Family-oriented scenario catalog with links to source configs, maps, benchmark surfaces, and caveats
* **[Hazard Traceability](./hazard_traceability.md)** - `hazard_traceability.v1` schema, typed loader, fixture, and coverage summary for scenario-to-hazard evidence caveats
* **[ODD Contracts](./odd_contracts.md)** - `odd_contract.v1` schema, typed loader, fixture, and boundary for benchmark and falsification evidence assumptions
* **[Simulation-Evidence Safety Case Template](./simulation_evidence_safety_case.md)** - Public-safe scaffold for mapping Robot SF simulation evidence, provenance, credibility gaps, and outside-simulation evidence requirements without implying certification
* **[Simulation Model Credibility Checklist](./context/issue_3290_simulation_model_credibility_checklist.md)** - Compact verification-vs-validation checklist and template for deciding whether simulation campaign evidence can support a stated benchmark or release claim
* **[Scenario Contracts](./scenario_contracts.md)** - `scenario_contract.v1` schema, typed loader, fixture, and boundary between authored intent, certification, and benchmark evidence
* **[Scenario Certification](./scenario_certification.md)** - `scenario_cert.v1` schema, CLI, labels, and fail-closed benchmark eligibility rules
* **[Scenario Perturbation Manifest](./scenario_perturbation_manifest.md)** - `scenario_perturbation_manifest.v1` schema, no-op and bounded route-offset preflight, and evidence boundary for perturbation pilots
* **[Issue #1272 Safety-Oriented Validation And Falsification Strategy](./context/issue_1272_validation_falsification_strategy.md)** - Current roadmap note for positioning Robot SF as validation/falsification infrastructure without certification or proof-of-safety claims
* **[Benchmark: Camera-ready / Scenario Reports](./benchmark_camera_ready.md)** - Camera-ready campaign workflow, planner report partitions, and publication-grade artifact contract
* **[Benchmark Suite Catalog](./benchmark_suites/README.md)** - Named suite IDs, canonical commands, runtimes, eligible planners, and claim boundaries for smoke, nominal, stress, AMV, LiDAR, and adversarial surfaces
* **[Issue #1073 Robot SF Empirical-Expansion Gate](./context/issue_1073_empirical_expansion_gate_2026_06_08.md)** - June 8 checkpoint rule for deciding whether Robot SF can move beyond dissertation-floor examples into bounded empirical expansion
* **[Benchmark Static Dashboard](./benchmark_static_dashboard.md)** - Self-contained static HTML dashboard generation from camera-ready benchmark bundles
* **[Static Leaderboards](./leaderboards/README.md)** - Markdown leaderboard row contract and first evidence-bound smoke, nominal-sanity, AMV, and LiDAR result surfaces
* **[Trajectory Debug Visualization](./debug_visualization.md)** - Optional JSON/Rerun timeline export for inspecting one episode JSONL without treating debug artifacts as benchmark evidence
* **[Learned-Policy Cards](./policy_cards/README.md)** - Human-readable learned-policy summaries that separate existence, smoke proof, benchmark comparison, and promotion boundaries
* **[PR Promoted Planner Smoke](./benchmark_pr_promoted_planner_smoke.md)** - Pull-request micro-benchmark workflow, runtime target, and fail-closed summary contract
* **[Issue #1065 Route-Clearance Warning Audit](./context/issue_1065_route_clearance_warning_audit.md)** - Paper and h500 route-clearance warning classification, planner-attribution boundary, and repair/certification follow-up
* **[Issue #595 Seed-Variability Contract](./context/issue_595_seed_variability_contract.md)** - Frozen camera-ready artifact contract and pilot slice for paper-side seed variability analysis
* **[Issue #832 Paper-Matrix Extended Seed Schedule](./context/issue_832_paper_matrix_extended_seed_schedule.md)** - Staged S5/S10/S20 seed extension policy, runtime estimates, tmux commands, and comparison artifact contract for the frozen paper matrix
* **[Issue #821 Paper Evidence Upgrade](./context/issue_821_paper_evidence_upgrade.md)** - Extended camera-ready matrix with guarded PPO, TEB, and SNQI ranking ablation evidence
* **[Issue #750 Paper Results Handoff](./context/issue_750_paper_results_handoff.md)** - Deterministic paper-facing JSON/CSV handoff contract for frozen benchmark bundles with CI metadata and provenance fields
* **[Benchmark Release 0.0.2 Execution Log](./context/benchmark_release_0_0_2_2026-04-13.md)** - All-planners release manifest/config decisions, tmux execution path, assumptions, and follow-up publish steps
* **[Benchmark Release 0.0.2 Publication Snapshot](./experiments/publication/20260414_benchmark_release_0_0_2/summary.md)** - Durable scoped seven-planner release pointer with DOI, archive checksum, embedded manifest/checksum/SNQI paths, and fresh-checkout recovery command
* **[Benchmark Release 0.0.2 Reproduction Note](./benchmark_release_0_0_2_reproduction.md)** - Dedicated copy-paste procedure for reproducing release 0.0.2 results from a fresh clone of main using the downloaded release archive and running the regeneration parity test
* **[Issue #1062 Paper Evidence Archive Pointer](./context/issue_1062_paper_evidence_archive.md)** - Paper evidence archive recovery note for the scoped `0.0.2` bundle without committing raw benchmark outputs

* **[Issue #435 Map Coverage Flow](./context/issue_435_map_coverage_flow.md)** - Parent flow state for map coverage, SocNavBench import, and map-quality child issues
* **[Issue #328 Real-World Map Parent Tracker](./context/issue_328_real_world_map_parent.md)** - Parent/child map-coverage split, child issue status, and shared validation contract for real-world benchmark maps
* **[Issue #692 Scenario Difficulty Analysis](./context/issue_692_scenario_difficulty_analysis.md)** - Artifact-driven camera-ready workflow for consensus ranking, planner residuals, and verified-simple subset assessment
* **[Issue #691 Benchmark Fallback Policy](./context/issue_691_benchmark_fallback_policy.md)** - Canonical fail-closed rule for fallback, degraded, and not-available benchmark outcomes
* **[Issue #1436 CI Reproducibility and Flaky Statistical Acceptance Policy (2026-05-22)](./context/issue_1436_reproducibility_flaky_acceptance.md)** - Canonical validation lanes, deterministic vs stochastic failure classification, and explicit CI rerun rules
* **[Issue #1360 External TEB Reference Assessment](./context/issue_1360_external_teb_assessment.md)** - Source-reuse and adapter-boundary assessment for external TEB-style corridor-deadlock baselines
* **[Issue #736 Station-Platform Candidate Pack](./context/issue_736_station_platform_candidate_pack.md)** - Exploratory station-platform variants, canonical commands, and conservative promotion boundary
* **[Issue #735 Platform Semantics Boundary](./context/issue_735_platform_semantics.md)** - Scenario-side platform hazard and keep-clear metadata contract, with fail-closed behavior for unsupported consumers
* **[Issue #717 Safety Barrier Spike](./context/issue_717_safety_barrier_spike.md)** - Clean-room native planner spike results showing the current heuristic runs but fails the verified-simple static slice
* **[Issues #717/#718 Safety Barrier Spike](./context/issue_717_safety_barrier_spike.md)** - Clean-room native planner spike and nominal-controller closeout; current static-slice evidence remains testing-only, not a promotion claim
* **[Grid Route Deep Dive](./context/issue_717_grid_route_deep_dive.md)** - Standalone experimental-planner note covering `grid_route` contract, full scenario-set deep dive, and the remaining `narrow_passage` boundary
* **[Issue #596 Atomic Scenario Suite Proposal](./context/issue_596_verified_simple_gate_proposal.md)** - Full-breadth atomic suite, verified-simple subset, validation fixtures, and scenario-contract rationale
* **[Issue #596 Atomic Scenario Matrix](./context/issue_596_atomic_scenario_matrix.md)** - Compact scenario-by-scenario matrix covering capabilities, failure modes, and verified-simple membership
* **[Issue #596 ORCA Failure Analysis](./context/issue_596_orca_failure_analysis.md)** - Targeted ORCA probe results showing which atomic scenarios still fail and why
* **[Issue #596 Testing-Only Planner Promotion Matrix](./context/issue_596_testing_only_planner_promotion_matrix.md)** - Planner-specific promotion blockers, evidence links, and next-proof requirements for the testing-only planners
* **[Benchmark Release Protocol v0.1](./benchmark_release_protocol.md)** - Canonical benchmark release model, versioning policy, and manifest/entrypoint contract for paper-facing releases
* **[Benchmark Release Reproducibility](./benchmark_release_reproducibility.md)** - Reproduce a benchmark release from a tag, canonical manifest, and reduced smoke validation path
* **[Benchmark Docker Reproduction Path](./benchmark_docker_repro.md)** - Build a pinned Docker image and run the canonical small benchmark artifact smoke with one command
* **[Camera-ready Release Workflow](./benchmark_camera_ready_release.md)** - Guided release upload checklist for campaign publication bundles
* **[Benchmark Observation Visibility](./benchmark_observation_visibility.md)** - Configurable planner-facing FOV, range, and static-occlusion filtering for partial-observability experiments
* **[Issue #1124 Dynamic Pedestrian Occlusion Contract](./context/issue_1124_dynamic_pedestrian_occlusion_contract.md)** - Opt-in planner-facing pedestrian-to-pedestrian occlusion semantics, metadata behavior, and benchmark-reporting limits
* **[Benchmark Planner-Family Coverage Matrix](./benchmark_planner_family_coverage.md)** - Benchmark-facing mapping from current planner/config support to Alyassi-style planner families, including readiness and overclaim guardrails
* **[Benchmark: Experimental Planners](./benchmark_experimental_planners.md)** - Opt-in guardrails and usage notes for unfinished benchmark planner families
* **[Planner Adapter Starter Template](./dev/planner_adapter_template.md)** - Copy-and-adapt path plus a diagnostic reference adapter for new local planner contributions
* **[Issue #589 Public Leaderboard MVP Boundary](./context/issue_589_public_leaderboard_mvp.md)** - Conservative no-implementation-now decision and future PR-based leaderboard prerequisites
* **[Issue #1086 Docker Reproduction Path](./context/issue_1086_docker_reproduction_path.md)** - Decision record and validation boundary for the pinned Docker benchmark smoke path
* **[Issue #1087 Planner Adapter Starter Template](./context/issue_1087_planner_adapter_template.md)** - Decision record for the reference adapter, docs, and validation path
* **[Policy Search Context](./context/policy_search/README.md)** - File-based local policy-search workflow with candidate registry, staged evaluation funnel, emitted reports, and SLURM handoff notes for expensive follow-up work
* **[Issue #1357 Tentabot-Style Motion-Primitive Assessment](./context/policy_search/2026-05-20_tentabot_motion_primitive_assessment.md)** - Source-backed verdict that Tentabot-style learned primitive-value scoring is a Robot SF-native spike candidate, not an upstream adapter or benchmark-ready planner
* **[Issue #1023 Scenario-Horizon Benchmark Surface](./context/issue_1023_scenario_horizon_benchmark.md)** - Runnable h500 scenario-horizon benchmark config, local non-Slurm full campaign evidence, fixed-vs-scenario comparison, and conservative promotion boundary
* **[Issue #1023 Experimental Benchmark Candidates](./context/issue_1023_experimental_benchmark_candidates.md)** - Rationale and caveats for adding `scenario_adaptive_hybrid_orca_v1` and `hybrid_rule_v3_fast_progress_static_escape` to the long-horizon benchmark as experimental challengers
* **[H500 Policy-Search Evidence Bundle](./context/evidence/policy_search_h500_2026-05-06/README.md)** - Durable policy-search evidence behind the h500 scenario-horizon schedule
* **[Issue #1023 Scenario-Horizon Preflight Evidence Bundle](./context/evidence/issue_1023_scenario_horizons_preflight_2026-05-06/README.md)** - Compact preflight proof for the paper-facing scenario-horizon benchmark matrix
* **[Issue #1023 Candidate-Augmented Preflight Evidence Bundle](./context/evidence/issue_1023_candidate_augmented_preflight_2026-05-06/README.md)** - Compact preflight proof for the 9-planner long-horizon matrix with the two experimental candidates
* **[Issue #1023 Candidate-Augmented Local Full Evidence Bundle](./context/evidence/issue_1023_candidate_augmented_local_full_2026-05-06/README.md)** - Compact 9-planner local full-campaign reports showing both experimental candidates execute, with SNQI release caveat
* **[Issue #1023 Scenario-Horizon Local Full Evidence Bundle](./context/evidence/issue_1023_scenario_horizons_local_full_2026-05-06/README.md)** - Compact local full-campaign reports, analyzer output, and fixed-vs-scenario comparison for the paper-facing scenario-horizon matrix
* **[Issue #1059 Deferred Planner-Improvement Program](./context/issue_1059_deferred_planner_improvement_program.md)** - Trace-to-child routing for the deferred h500 planner-improvement program, including #1034 targeted recovery evidence and the #1113 full-matrix boundary
* **[Issue #1074 Robot-SF Worked-Example Pack](./context/issue_1074_robot_sf_worked_example_pack.md)** - Three retained h500 examples mapped to scenario class, actor mix, metric layer, failure-pattern vocabulary, and explicit claim boundaries
* **[Issue #1075 Operating Envelope And Non-Claims](./context/issue_1075_operating_envelope.md)** - Canonical Robot-SF dissertation-floor evidence envelope, supported evidence types, non-claims, and future-work boundaries
* **[Issue #1083 Sanity V1 Nominal Matrix](./context/issue_1083_sanity_v1_nominal_matrix.md)** - Non-paper-facing nominal calibration matrix and smoke command for easier deployment-like scenes
* **[Issue #1082 Paper Cross-Kinematics Parity Sweep](./context/issue_1082_paper_cross_kinematics_v1.md)** - Versioned paper-facing cross-kinematics smoke profile, compatibility manifest, and interpretation boundary
* **[Issue #1274 General Cross-Kinematics Parity Sweep](./context/issue_1274_cross_kinematics_v1.md)** - Non-paper cross-kinematics parity profile, compatibility manifest, and interpretation boundary
* **[Issue #1084 Planner Inclusion Gate](./context/issue_1084_planner_inclusion_gate.md)** - Mechanical `planner-inclusion-check` command, report schema, thresholds, and pass/revise proof cases for promotion review
* **[Benchmark Mechanism Roadmap Plan (2026-05-07)](./superpowers/plans/2026-05-07-benchmark-mechanism-roadmap.md)** - Agent-executable plan for the trace-backed h500 mechanism pilot, deferred planner-improvement capture, and deferred CARLA-transfer capture
* **[Goal Sequence Spec (2026-05-07)](./superpowers/specs/2026-05-07_goal_sequence.md)** - Issue sequencing rationale that keeps paper evidence first, h500 mechanism interpretation next, and deferred strategic alternatives explicit
* **[Issue #872 CARLA Oracle Replay Bridge Status](./context/issue_872_carla_oracle_replay_bridge_status.md)** - Closure record for the bounded CARLA replay/parity parent and the current setup/adapted/native/metric-parity claim boundaries
* **[Issue #1485 CARLA Transfer-Boundary Follow-Up](./context/issue_1485_carla_transfer_boundary_follow_up.md)** - Post-closure CARLA transfer-boundary taxonomy, deferred multi-scenario replay boundary, and artifact-discipline reminder
* **[Issue #1508 CARLA Native/Aligned Eligibility Audit](./context/issue_1508_carla_native_aligned_eligibility.md)** - Pre-campaign table classifying certified, native-spawn, and native-metric CARLA replay candidates before any multi-scenario launch
* **[Issue #1444 CARLA Coordinate Alignment Contract (2026-05-22)](./context/issue_1444_carla_coordinate_alignment_contract.md)** - Conservative replay-mode taxonomy and projection tolerances required before any Robot-SF/CARLA metric parity claim
* **[Issue #1110 CARLA Oracle Replay Parity Adapter](./context/issue_1110_carla_oracle_replay_parity_adapter.md)** - Conservative metric-parity report format, fail-closed degraded-mode handling, and CLI validation boundary before live CARLA evidence exists
* **[Camera-Ready All-Planners SLURM Check (2026-05-04)](./context/camera_ready_all_planners_slurm_2026-05-04.md)** - Failed `rsf-allbench` job, partial all-planners evidence, asset blocker, and rerun boundary
* **[SocNav Asset Setup (License-Safe)](./socnav_assets_setup.md)** - Official-source download/staging instructions for SocNav third-party datasets with validation commands
* **[Benchmark Runner & Metrics](./benchmark.md)** - Episode schema, aggregation, metrics suite (collisions, comfort exposure, SNQI), and validation hooks
* **[Issue #1434 Stress/Uncertainty Coverage Schema v1](./context/issue_1434_stress_uncertainty_coverage_schema.md)** - `stress_uncertainty_coverage.v1` field contract, statistical summary tiers, coverage axes, interpretation boundaries, and fail-closed consumer rules for benchmark reports
* **[Full Classic Interaction Benchmark](./benchmark_full_classic.md)** - Complete guide: episodes, aggregation, effect sizes, adaptive precision, plots, videos, scaling metrics
* **[Benchmark Artifact Publication](./benchmark_artifact_publication.md)** - Public artifact policy, DOI-ready export bundles, release/Zenodo workflow
* **[Artifact Catalog v1](./artifact_catalog.md)** - Stable semantic IDs, checksums, generation commands, and claim boundaries for reusable figures and tables
* **[Multi-AMV Benchmark First Slice](./multi_amv_benchmark.md)** - Minimal multi-robot scenario surface, validation smoke, and inter-robot metric block
* **[Issue #1128 Multi-AMV Episode Extension](./context/issue_1128_multi_amv_episode_extension.md)** - Additive multi-AMV episode block, canonical `metrics.inter_robot` JSONL/report output, and fail-closed validation notes
* **[Issue #1168 Multi-AMV Planner Support Classification](./context/issue_1168_multi_amv_planner_support.md)** - Planner-family support inventory, fail-closed multi-AMV preflight gate, and the boundary between smoke control and real multi-robot planner support
* **[Real-World Trajectory Import](./real_world_trajectory_import.md)** - Narrow Stanford Drone Dataset annotation importer, normalization contract, and provenance workflow
* **[Benchmark Visual Artifacts](./benchmark_visuals.md)** - SimulationView & synthetic video pipeline, performance metrics
* **[Metrics Specification](./dev/issues/social-navigation-benchmark/metrics_spec.md)** - Formal definitions of benchmark metrics (includes per-pedestrian force quantiles)
* **[Local Navigation Benchmark Gap Analysis (2026-01-14)](./dev/benchmark_plan_2026-01-14.md)** - Current-state inventory, missing pieces, and open questions for local planner benchmarking
* **[Prediction Planner Baseline](./baselines/prediction_planner.md)** - High-level model description, benchmark role, configuration, and citation/provenance notes
* **[Prediction-Aware MPC Planner](./baselines/prediction_mpc.md)** - Experimental constant-velocity prediction-MPC local planner, config, smoke command, and claim boundary
* **[Prediction Planner Literature Audit](./context/prediction_planner_literature_audit.md)** - Source-backed audit of implementation lineage, benchmark evidence, literature-positioning boundaries, and current claim limits
* **[Issue #592 Hybrid Obstacle-Context Predictor Design](./context/issue_592_hybrid_obstacle_predictor_design.md)** - Feature-baseline-first plan for obstacle-conditioned predictive models, with config-first experiment path and benchmark proof gates
* **[Guarded PPO Baseline](./baselines/guarded_ppo.md)** - Canonical safety-aware challenger profile, intervention semantics, and benchmark-readiness boundary
* **[Prediction Planner PR Readiness (2026-02-20)](./context/predictive_planner_pr_readiness_2026-02-20.md)** - Completed integration checklist and remaining maintainer decisions before final merge
* **[Issue #454 Execution Note (Kinematics Contract + Parity)](./context/issue_454_kinematics_parity_execution_note.md)** - Runtime kinematics wiring, feasibility diagnostics, and parity campaign evidence summary
* **[Issues 485-492 Execution Trace](./context/issues_485_492_execution.md)** - Implementation summary, validation runs, and rollout notes for the benchmark hardening changes
* **[Issue 499 Execution Notes](./context/issue_499_execution.md)** - Publication bundle tooling, policy, and size-measurement workflow
* **[Issues 500-504 Execution Notes](./context/issues_500_501_504_execution.md)** - Metadata enrichment, time-to-goal contract extensions, and adapter-impact probing implementation
* **[Benchmark Post-Prediction Fix Report (2026-02-20)](./context/benchmark_post_prediction_fix_2026-02-20.md)** - Baseline promotion, predictive compatibility fixes, hotspot diagnostics, and campaign comparison artifacts
* **[Issue #535 Execution Note (Paper Matrix Freeze)](./context/issue_535_paper_matrix_freeze.md)** - Frozen paper-facing experiment matrix contract and canonical command path
* **[Issue #600 DSRNN Stretch Follow-up](./context/issue_600_dsrnn_stretch_follow_up.md)** - Canonical DSRNN source anchors, dependency ordering behind the first attention/prediction spikes, and the extra graph/recurrent integration burden that keeps this family in assessment-only status
* **[Issue #603 Alyassi Planner Reference Set](./context/issue_603_alyassi_reference_set_2026-03-06.md)** - Canonical paper-side planner anchors, citekeys, local clone locations, and current SoNIC provenance caveats
* **[Issue #651 Social-Navigation-PyEnvs ORCA Self-Velocity Contract Note](./context/issue_651_social_navigation_pyenvs_orca_self_velocity.md)** - Explicit planar self-velocity contract fix for the upstream ORCA adapter, restored raw `ActionXY` parity on upstream scenarios, and paper-surface rerun showing the corrected prototype remains weak
* **[Issue #653 Social-Navigation-PyEnvs SocialForce Runtime Reproduction Note](./context/issue_653_social_navigation_pyenvs_socialforce_runtime.md)** - Compatibility-runtime probe proving the upstream SocialForce policy can run unchanged against `socialforce==0.2.3` through an explicit shim, which justifies a benchmark-facing retry
* **[Issue #656 Social-Navigation-PyEnvs SocialForce Retry Note](./context/issue_656_social_navigation_pyenvs_socialforce_retry.md)** - Benchmark-facing SocialForce retry showing the constructor blocker is fixed through an explicit shim, but the planner remains weak and partially unstable on the paper surface
* **[Issue #659 gym-collision-avoidance Headless Reproduction](./context/issue_659_gym_collision_avoidance_headless.md)** - Explicit headless compatibility shims that let the upstream GA3C-CADRL example complete in the isolated side environment, turning this family from generic runtime-blocked into wrapper/parity-justified
* **[Issue #661 gym-collision-avoidance Model Parity Probe](./context/issue_661_gym_collision_avoidance_model_parity.md)** - Live native-observation parity check showing the current local `_SACADRLModel` already matches upstream GA3C-CADRL checkpoint inference, shifting the remaining risk to Robot SF observation mapping and benchmark behavior
* **[Issue #663 SACADRL Observation Parity Note](./context/issue_663_sacadrl_observation_parity.md)** - Controlled and live upstream parity probe showing the current `SACADRLPlannerAdapter` reproduces the GA3C-CADRL network input on tested cases, shifting the remaining risk from observation mapping to planner quality and scenario transfer
* **[Issue #690 Holonomic Benchmark Feasibility Note](./context/issue_690_holonomic_benchmark_feasibility.md)** - Parallel holonomic benchmark sibling, strict fail-closed planner policy, and adapter-strategy feasibility summary
* **[Issue #759 Francis Guideline Mapping](./context/issue_759_francis_guideline_mapping.md)** - Mapping Robot SF benchmark contract to Francis et al. social navigation evaluation principles
* **[Issue #649 Social-Navigation-PyEnvs ORCA Parity Note](./context/issue_649_social_navigation_pyenvs_orca_parity.md)** - Same-snapshot upstream-versus-wrapper parity probe showing the pre-fix self-velocity contract mismatch in the original ORCA adapter before downstream unicycle projection
* **[Issue #647 Social-Navigation-PyEnvs HSFM Integration Note](./context/issue_647_social_navigation_pyenvs_hsfm_integration.md)** - Benchmark-facing headed-force-model prototype for upstream HSFM-New-Guo with explicit body-frame action contract, angular-rate observation extension, and benchmark validation path
* **[Issue #646 Social-Navigation-PyEnvs SocialForce and SFM Integration Note](./context/issue_646_social_navigation_pyenvs_force_models_integration.md)** - Benchmark-facing force-model split showing SFM-Helbing runs as an upstream-backed sanity-surface prototype while SocialForce is blocked by an external package API mismatch
* **[Issue #644 Social-Navigation-PyEnvs ORCA Integration Note](./context/issue_644_social_navigation_pyenvs_orca_integration.md)** - Benchmark-facing prototype planner entry for upstream Social-Navigation-PyEnvs ORCA with explicit provenance and projection contract, plus a successful proof campaign and conservative claim boundary
* **[Issue #642 Social-Navigation-PyEnvs Source Harness Probe](./context/issue_642_social_navigation_pyenvs_source_harness_probe.md)** - Gymnasium-native source-harness and wrapper probe showing simulator-core viability, shimmed upstream ORCA reset/step proof, and a real Robot SF wrapper-loop smoke path with explicit compatibility boundaries
* **[Issue #641 gym-collision-avoidance Side-Environment Reproduction](./context/issue_641_gym_collision_avoidance_side_env.md)** - Isolated legacy-runtime reproduction showing GA3C-CADRL learned-policy initialization succeeds in a side environment while the canonical upstream example remains blocked on the macOS TkAgg visualization path
* **[Issue #639 gym-collision-avoidance Source Harness Probe](./context/issue_639_gym_collision_avoidance_source_harness_probe.md)** - Fail-fast CADRL-family source-harness probe, extracted observation/action contract, and blocked verdict showing wrapper work is not yet justified in the main runtime
* **[Issue #635 SNQI v3 Paper-Facing Contract Note](./context/issue_635_snqi_v3_paper_contract.md)** - Exact SNQI v3 asset contract, delta versus the corrected pre-v3 rerun, canonical field mapping, and paper regeneration guidance
* **[Issue #838 SNQI Calibration Analysis](./context/issue_838_snqi_calibration_analysis.md)** - Follow-up sensitivity analysis comparing fixed v3 against weight perturbations and normalization-anchor variants
* **[Issue #632 Python-RVO2 Prototype Note](./context/issue_632_python_rvo2_prototype.md)** - Upstream ORCA provenance, explicit velocity-to-unicycle projection contract, validation commands, and same-surface comparison showing documentation gain but no paper-matrix performance upgrade
* **[Issue #605 gym-collision-avoidance Reference Note](./context/issue_605_gym_collision_avoidance_reference_note.md)** - CADRL-family reference assessment, source-harness recommendation, and explicit wrapper boundary for future reproduction
* **[Issue #604 Pred2Nav Assessment Note](./context/issue_604_pred2nav_assessment.md)** - External predictive-MPC assessment showing Pred2Nav is useful as a concept source but blocked for direct reuse by unclear licensing, legacy runtime, and holonomic action semantics
* **[Issue #599 Go-MPC Assessment Note](./context/issue_599_go_mpc_assessment.md)** - External prediction-based planner-family assessment showing Go-MPC is solver-locked, GPL, and a poor direct integration target relative to current native predictive planners
* **[Issue #3985 ACMPC Feasibility Assessment](./context/issue_3985_acmpc_feasibility_assessment.md)** - Actor-Critic Model Predictive Control scoping note for a possible Robot SF-native learned-MPC planner, with adapter-burden, benchmark-boundary, and conditional design-child recommendation
* **[Issue #593 Predictive Ego-Conditioned v2 Note](./context/issue_593_predictive_ego_conditioned_v2.md)** - Collector-parity closeout for the 9D ego-conditioned predictive-planner path, plus staged evidence showing much better ADE/FDE but no hard-seed outcome gain yet
* **[Issue #591 Prediction Planner Probabilistic Search](./context/issue_591_prediction_planner_probabilistic_search.md)** - Probabilistic CVaR and MCTS-lite predictive modes improve safety/SNQI slightly but reduce success and impose a large runtime penalty, so they remain experimental only
* **[Issue #669 Prediction Planner v2 Benchmark Comparison](./context/issue_669_predictive_v2_benchmark_comparison.md)** - Hard-seed and full paper-surface comparison showing the ego-conditioned v2 checkpoint is only a mild tradeoff improvement over the current predictive baseline, not a headline planner-quality win
* **[Issue #1856 Predictive-v2 Coupling Objective](./context/issue_1856_predictive_coupling_objective.md)** - Proposal/preflight note for testing a phase-coupled planner objective and closed-loop success gate before any renewed four-way predictive-v2 expansion
* **[Issue #1897 Predictive Coupling Gate Preflight](./context/issue_1897_predictive_coupling_gate_preflight.md)** - Local closed-loop gate execution showing the phase-coupled row only improved clearance, not success, so predictive-v2 expansion stays blocked
* **[Issue #671 Gap Prediction Benchmark Note](./context/issue_671_gap_prediction_benchmark.md)** - Paper-surface comparison showing `gap_prediction` is safer and faster than the predictive baselines but completely fails on goal-reaching, so it remains testing-only
* **[Issue #673 Hybrid Portfolio Benchmark Note](./context/issue_673_hybrid_portfolio_benchmark.md)** - Paper-surface comparison showing `hybrid_portfolio` is slower and weaker than both predictive baselines, so it remains testing-only
* **[Issue #675 Predictive MPPI Benchmark Note](./context/issue_675_predictive_mppi_benchmark.md)** - Paper-surface comparison showing `predictive_mppi` is much slower and weaker than the predictive baselines, so it remains testing-only
* **[Issue #677 MPPI Social Benchmark Note](./context/issue_677_mppi_social_benchmark.md)** - Paper-surface comparison showing `mppi_social` trades some safety signal for much worse runtime and goal-reaching, so it remains testing-only
* **[Issue #679 Risk DWA Benchmark Note](./context/issue_679_risk_dwa_benchmark.md)** - Paper-surface comparison showing `risk_dwa` is faster and has fewer near misses but loses too much success and collision quality to be promoted
* **[Issue #681 Stream Gap Benchmark Note](./context/issue_681_stream_gap_benchmark.md)** - Paper-surface comparison showing `stream_gap` is extremely safe and fast but fails completely on goal-reaching, so it remains testing-only
* **[Issue #684 Guarded PPO Tuning](./context/issue_684_guarded_ppo_tuning.md)** - Config-only guarded-PPO threshold/fallback tuning showing the best relaxed profile recovers some success but still falls well short of the benchmark leaders
* **[Issue #768 ORCA Variants Benchmark](./context/issue_768_orca_variants.md)** - ORCA variant evaluation (nonholonomic, DD-style, relaxed, HRVO) on classic interactions, with a conservative decision to promote `socnav_orca_dd`
* **[Issue #602 Guarded PPO Safety-Aware Profile Note](./context/issue_602_guarded_ppo_profile.md)** - Canonical guarded-PPO contract, paper-surface comparison against `ppo` and `orca`, and conservative boundary for internal safety-aware support
* **[Issue #629 Planner Zoo Research Prompt](./context/issue_629_planner_zoo_research_prompt.md)** - Deep-research prompt, evaluation rubric, and execution sequence for external local planner candidates
* **[External Planner Reuse Checklist](./context/external_planner_reuse_checklist.md)** - Fail-fast upstream provenance, source-harness repro, wrapper smoke, and verdict checklist for future planner reuse work
* **[Issue #626 SoNIC Source Harness Probe](./context/issue_626_sonic_source_harness_probe.md)** - Fail-fast source-harness reproduction command, captured SoNIC contract, blocked verdict, and model-only reuse follow-up
* **[Issue #627 SoNIC Wrapper Follow-up](./context/issue_627_sonic_wrapper_followup.md)** - Fail-fast model-only Robot SF wrapper, translation tests, benchmark-boundary verdict, and current source-harness limitation
* **[Issue #601 CrowdNav Feasibility Note](./context/issue_601_crowdnav_feasibility_note.md)** - Family assessment, canonical source anchors, and integration shape decision for CrowdNav attention-based crowd navigation
* **[Issue #695 `safe_control` Feasibility Note](./context/issue_695_safe_control_feasibility_note.md)** - External safety-controller assessment showing the current path is blocked by missing runtime dependencies, unclear license metadata, and a waypoint-tracking contract mismatch
* **[Issue #581 Paper Evidence Delta Report](./context/issue_581_paper_evidence_delta.md)** - Canonical corrected benchmark artifact handoff, planner-quality claim boundary, and AMV paper-ingestion checklist
* **[Issue #1236 Optimizer Adversarial Sampler](./context/issue_1236_optimizer_adversarial_sampler.md)** - Optuna-backed adversarial sampler pilot, synthetic comparison helper, and non-paper-facing evidence boundary
* **[Issue #1271 Seed-Sensitivity Explorer](./context/issue_1271_seed_sensitivity_explorer.md)** - API and summary contract for replaying adversarial candidates over explicit seed grids while keeping failure persistence and artifact claims bounded
* **[Issue #1294 Seed-Sensitivity Perturbations](./context/issue_1294_seed_sensitivity_perturbations.md)** - Opt-in timing/speed perturbation grid for seed-sensitivity replays, with bounded deltas and explicit non-benchmark evidence limits
* **[Issue #1609 Seed-Sensitive Scenario Mechanisms](./context/issue_1609_seed_sensitive_mechanisms.md)** - Diagnostic mechanism hypotheses over Issue #1608 seed-sensitive scenarios with hard-vs-easy aggregate tables and trace-review limits
* **[Planner Quality Audit Workflow](./benchmark_planner_quality_audit.md)** - Build the planner decision table, classify headline suitability, and record paper-faithfulness parity gaps

### Tooling

* **[SNQI Weight Tools](./snqi-weight-tools/README.md)** - Recompute, optimize, and analyze SNQI weights; command reference and workflow examples
* **[Runtime Requirements Checker](./dev_runtime_requirements.md)** - Inventory non-Python host tools with `scripts/dev/check_runtime_requirements.sh`
* **[Pyreverse UML Generation](./pyreverse.md)** - Generate class diagrams from code
* **[Data Analysis Utilities](./DATA_ANALYSIS.md)** - Analysis helpers and data processing tools
* **[Imitation Results Analysis](./imitation_results_analysis.md)** - Compare baseline vs pre-trained runs, emit training summaries and figures
* **[SVG Inspection Workflow](./dev/svg_inspection_workflow.md)** - Inspect route/zone consistency, parser-risky path commands, and obstacle crossings with `scripts/validation/svg_inspect.py`

### Architecture & Refactoring

* **[Refactoring Overview](./refactoring/)** - Complete guide to the refactored environment architecture (deployment status, plan, migration guide, summary, automated codebase analysis)
* **[Subtree Migration Guide](./SUBTREE_MIGRATION.md)** - Git subtree integration for fast-pysf (migration from submodule)
* **[UV Migration Notes](./UV_MIGRATION.md)** - Migration to UV package manager
* **[Repository Structure Analysis](./dev/issues/repository-structure-analysis.md)** - Comprehensive assessment of codebase organization and improvement roadmap
* **[Agents & Contributor Onboarding](../AGENTS.md)** - High-level repository structure, coding/testing conventions, workflow tips
* **[Acknowledgments & Upstream Work](../ACKNOWLEDGMENTS.md)** - Preserved provenance, cited papers, and upstream repository lineage for the project
* **[Benchmark/Planner Review Guide](./code_review.md)** - Review checklist for benchmark semantics, normalization, scenario distributions, reproducibility, and provenance

### Simulation & UI

* **[Simulation View](./SIM_VIEW.md)** - Visualization and rendering system
* **[LiDAR Configuration Reference](./lidar_configuration.md)** - Canonical robot and ego-pedestrian scan defaults, including ray count, field of view, range, and noise
* **[Helper Catalog](./dev/helper_catalog.md)** - Reusable render helpers for frame capture, output directories, and video contact sheets
* **[SVG Map Editor](./SVG_MAP_EDITOR.md)** - SVG-based map creation tools and usage
* **[OSM Map Generation](./osm_map_workflow.md)** - Programmatic, reproducible maps from OpenStreetMap data (PBF import, zone/route definition, scenario creation)
  + **Quick Start**: 3 approaches (visual editor, programmatic API, hybrid)
  + **API Reference**: 6 helper functions (zones, routes, config management, YAML loading)
  + **Examples**: 4 realistic scenarios (simple navigation, urban intersection, variable density, load/verify)
* **[Map Verification](../specs/001-map-verification/quickstart.md)** - Validate SVG maps for structural integrity and runtime compatibility
* **[Issue 388 Execution Notes](./context/issue_388_execution.md)** - Self-intersecting obstacle-path repair behavior and validation details
* **[Francis 2023 Scenario Pack](../maps/svg_maps/francis2023/readme.md)** - SVG maps +
  scenario matrix for Fig. 7 archetypes; definitions in
  [configs/scenarios/francis2023.yaml](../configs/scenarios/francis2023.yaml)
* **[Occupancy Grid Guide](./dev/occupancy/Update_or_extend_occupancy.md)** - Configure grid observations, spawn queries, and pygame overlays
* **[Circle Rasterization Fix](./dev/issues/circle-rasterization-fix/README.md)** - Clarifies circle overlap handling in occupancy grid rasterization
* **[Telemetry Pane & Headless Artifacts](../specs/343-telemetry-viz/quickstart.md)** - Enable docked charts in Pygame, replay/export telemetry, and run headless smoke tests
* **[Telemetry Pane Display Fix](./telemetry-pane-fix.md)** - Technical analysis and solution for continuous graph rendering, surface caching, and buffer management

### Figures & Visualization

* **[Trajectory Visualization](./trajectory_visualization.md)** - Generate trajectory plots
* **[Force Field Visualization](./force_field_visualization.md)** - Heatmap + quiver figures (PNG/PDF)
* **[Pareto Plotting](./pareto_plotting.md)** - Generate Pareto frontier plots
* **[Planner Tradeoff Plotting](./planner_tradeoff_plotting.md)** - Generate success/collision tradeoff figures from publication bundles
* **[Force Field Heatmap](./force_field_heatmap.md)** - Heatmap + vector overlays figure (PNG/PDF)

### Performance & CI

* **[Performance Notes](./performance_notes.md)** - Performance targets, benchmarking, and optimization notes
* **[Issue 2536 Speed Discovery](./context/issue_2536_speed_discovery.md)** - Bounded simulator-speed candidate discovery and next occupancy-grid rasterization proof path
* **[Issue 483 Execution Notes](./context/issue_483_execution.md)** - Cold/warm regression guard implementation details and workflow wiring
* **[Issue 495 Execution Notes](./context/issue_495_execution.md)** - Overall trend benchmark matrix, history comparison, and nightly cache-backed tracking
* **[Warning Hygiene Sweep](./context/warning_hygiene_2026-02-13.md)** - Warning-noise root-cause fixes and dependency mitigation notes
* **[Coverage Guide](./coverage_guide.md)** - Code coverage collection, baseline tracking, CI integration

### Hardware & Environment

* **[Environment Configuration](./ENVIRONMENT.md)** - Detailed environment setup and usage
* **[Runtime Requirements](./dev_runtime_requirements.md)** - Host tool, Docker, GPU, SLURM, and headless rendering requirements outside `uv`

---

### Additional Resources (Legacy Structure)

<details>
<summary>Click to expand legacy detailed index</summary>

### 🏗️ Architecture & Development

* **[Development Guide](./dev_guide.md)** - Primary reference for development workflows, testing, and quality standards
* **[Configuration Architecture](./architecture/configuration.md)** - Configuration hierarchy, precedence rules, and migration guide
* **[Repository Structure Analysis](./dev/issues/repository-structure-analysis.md)** - Comprehensive assessment of codebase organization and improvement roadmap
* **[Coverage Guide](./coverage_guide.md)** - Comprehensive guide to code coverage collection, baseline tracking, and CI integration
* **[Environment Refactoring](./refactoring/)** - **NEW**: Complete guide to the refactored environment architecture
  + [Deployment Status](./refactoring/DEPLOYMENT_READY.md) - Current implementation status
  + [Refactoring Plan](./refactoring/refactoring_plan.md) - Technical architecture details
  + [Migration Guide](./refactoring/migration_guide.md) - Step-by-step migration instructions
  + [Implementation Summary](./refactoring/refactoring_summary.md) - What was accomplished
  + [Migration Report](./refactoring/migration_report.md) - Automated codebase analysis
  + **Classic interactions refactor (Feature 139)** — Design note: Extract visualization & formatting helpers — `docs/dev/issues/classic-interactions-refactor/design.md`
* **[Architectural Decoupling (Feature 149)](../specs/149-architectural-coupling-and/)** - Backend and sensor registry system for extensible simulation
  + [Quickstart Guide](../specs/149-architectural-coupling-and/quickstart.md) - Usage examples for backend selection and sensor registration
  + [Tasks & Progress](../specs/149-architectural-coupling-and/tasks.md) - Implementation task tracking
* **[Agents & Contributor Onboarding](../AGENTS.md)** – High-level repository structure, coding/testing conventions, and workflow tips for new contributors

### 🎮 Simulation & Environment

* [**Simulation View**](./SIM_VIEW.md) - Visualization and rendering system
* [**SVG Map Editor**](./SVG_MAP_EDITOR.md) - SVG-based map creation tools
* [**Single Pedestrians**](./single_pedestrians.md) - Define individual pedestrians with goals or trajectories in SVG/JSON/code
* [**Multi-Pedestrian Example**](../examples/example_multi_pedestrian.py) - Demonstrates multiple single pedestrians (goal, trajectory, static) in one scenario
* [**Scenario Specification Checklist**](./scenario_spec_checklist.md) - Authoring checklist for per-scenario/archetype/manifest files
* [**Hazard Traceability**](./hazard_traceability.md) - Summarize intended hazard coverage for scenario IDs or families without treating traceability as safety proof
* [**ODD Contracts**](./odd_contracts.md) - Validate operating-assumption metadata that bounds benchmark and falsification evidence without certifying safety
* [**Simulation-Evidence Safety Case Template**](./simulation_evidence_safety_case.md) - Map benchmark artifacts to bounded safety-case sections while naming simulation limits and external evidence needs
* [**Scenario Contracts**](./scenario_contracts.md) - Validate authored scenario-intent contracts before certification or benchmark execution
* [**Scenario Certification**](./scenario_certification.md) - Generate machine-readable validity, feasibility, stress-only, and hard-but-solvable certificates for scenario manifests
* [**Issue #1240 Scenario Coverage Entropy**](./context/issue_1240_scenario_coverage_entropy.md) - Config-only entropy and novelty report for diagnostic scenario-set curation; not benchmark-success evidence
* **Classic Interaction Scenario Pack** (configs/scenarios/classic_interactions.yaml) – Canonical crossing, head‑on, overtaking, bottleneck, doorway, merging, T‑intersection, station-platform, and group crossing archetypes for benchmark coverage. See also [Issue #549 station-platform map rationale](./context/issue_549_station_platform_map.md).
* **[Francis 2023 Scenario Pack](../maps/svg_maps/francis2023/readme.md)** - SVG maps +
  scenario matrix in [configs/scenarios/francis2023.yaml](../configs/scenarios/francis2023.yaml).
* **Classic Interactions PPO Visualization (Feature 128)** – Deterministic PPO policy demo with optional recording (docs: `docs/dev/issues/classic-interactions-ppo/` | spec+plan+tasks under `specs/128-classic-interactions-ppo/`).

### 📊 Analysis & Tools

* [**SNQI Weight Tooling**](./snqi-weight-tools/README.md) - User guide for recomputing, optimizing, and analyzing SNQI weights
* [**SNQI Figures (orchestrator usage)**](../examples/README.md) - Generate SNQI-augmented figures from existing episodes
* [**Full SNQI Flow (episodes → baseline → figures)**](../examples/benchmarks/snqi_full_flow.py) - End-to-end reproducible pipeline script

* [**Benchmark Schema & Aggregation Diagnostics**](./benchmark.md) - Episode metadata mirrors, algorithm grouping keys,  `_meta` warnings, and validation hooks
* [Regression Notes – Algorithm Aggregation](./dev/issues/142-aggregation-mixes-algorithms/design.md) - Test matrix, warnings, and smoke workflow for Feature 142
* [**Social Navigation Benchmark**](./dev/issues/social-navigation-benchmark/README.md) - Benchmark design, metrics, schema, and how to run episodes/batches
* **Full Classic Interaction Benchmark** – Implementation complete (episodes, aggregation, effect sizes, adaptive precision, plots, videos, scaling metrics). See detailed guide: [ `benchmark_full_classic.md` ](./benchmark_full_classic.md) (quickstart & tasks in `specs/122-full-classic-interaction/` ).
* **Benchmark Visual Artifacts** – SimulationView & synthetic video pipeline, performance metrics: [ `benchmark_visuals.md` ](./benchmark_visuals.md)
* **Episode Video Artifacts (MVP)** – Design notes and links: [ `docs/dev/issues/video-artifacts/design.md` ](./dev/issues/video-artifacts/design.md)
* [**Baselines**](./dev/baselines/README.md) — Overview of available baseline planners
  + [Random baseline](./dev/baselines/random.md) — how to use and configure
* [**Force Field Visualization**](./force_field_visualization.md) — How to generate heatmap + quiver figures (PNG/PDF)
* [**Scenario Thumbnails & Montage**](./scenario_thumbnails.md) — Generate per-scenario thumbnails and montage grids (PNG/PDF)
* [**Planner Tradeoff Plotting**](./planner_tradeoff_plotting.md) — Generate safety-efficiency figures from publication bundles
* [**Force Field Heatmap**](./force_field_heatmap.md) — Heatmap + vector overlays figure (PNG/PDF)

</details>

#### Social Navigation Benchmark (Overview)

The benchmark layer provides:

* Deterministic episode JSONL schema (versioned) with per-episode metrics.
* Batch runner with resume manifest for incremental extensions.
* Metrics suite + SNQI composite index (with weight recomputation tooling).
* Aggregation + bootstrap CI utilities for statistical reporting.
* Figure orchestrator to generate distributions, Pareto frontiers, force-field visualizations, thumbnails, and tables.
  See the dedicated design page above for full specification and usage examples.

#### Figures naming and outputs

See `docs/dev/issues/figures-naming/design.md` for the canonical figure folder naming scheme and migration plan. Use `docs/artifact_catalog.md` when a generated figure or table needs a durable semantic ID that survives path regeneration. A small tracker lives at `docs/dev/issues/figures-naming/todo.md` .

#### LaTeX Table Embedding (SNQI / Benchmark Tables)

The figures orchestration script writes `baseline_table.md` by default. To obtain a LaTeX version suitable for direct inclusion:

1. Fast path: run the figures orchestrator with `--table-tex` to produce `baseline_table.tex` automatically.

```bash
uv run python scripts/generate_figures.py \
  --episodes output/benchmarks/episodes_sf_long_fix1.jsonl \
  --auto-out-dir --no-pareto --table-tex \
  --dmetrics collisions,comfort_exposure,snqi --table-metrics collisions,comfort_exposure,snqi
```

2. Alternative: use the CLI table command with `--format tex` for custom file naming:

```bash
uv run python -m robot_sf.benchmark.cli table \
  --episodes output/benchmarks/episodes_sf_long_fix1.jsonl \
  --metrics collisions,comfort_exposure,near_misses,snqi \
  --format tex > docs/figures/table_snqi.tex
```

3. Include in LaTeX:

```latex
\input{docs/figures/table_snqi.tex}
```

4. The output uses `booktabs`; ensure your preamble contains:

```latex
\usepackage{booktabs}
```

Optional tuning:

* Reorder metrics via `--metrics` list order.
* Confidence intervals (bootstrap):
  1. Produce an aggregate summary with bootstrap CIs:

```bash
     uv run robot_sf_bench aggregate \
       --in output/benchmarks/episodes_sf_long_fix1.jsonl \
       --out output/benchmarks/summary_ci.json \
       --bootstrap-samples 1000 --bootstrap-confidence 0.95 --bootstrap-seed 123
     ```

  2. Generate tables from the summary adding CI columns:

```bash
     uv run python scripts/generate_figures.py \
       --episodes output/benchmarks/episodes_sf_long_fix1.jsonl \
       --table-summary output/benchmarks/summary_ci.json \
       --table-metrics collisions,comfort_exposure,snqi \
       --table-stats mean,median,p95 \
       --table-include-ci --table-tex --no-pareto \
       --out-dir docs/figures/ci_example
     ```

  3. Column naming pattern in Markdown: `<metric>_<stat>` plus `<metric>_<stat>_ci_low` / `_ci_high` (or with a custom suffix if `--ci-column-suffix ci95` is used → `_ci95_low/_ci95_high`).
  4. LaTeX version escapes underscores automatically; just `\input{...}` as usual.
  5. Missing CI arrays (e.g., when a stat lacked bootstrap) trigger a consolidated warning and empty cells.

Available CI options:
* `--table-include-ci` add interval columns.
* `--ci-column-suffix ci95` change suffix (default `ci`).

Example (custom suffix for 90% CIs):

```bash
uv run robot_sf_bench aggregate \
  --in output/benchmarks/episodes.jsonl --out output/benchmarks/summary_ci90.json \
  --bootstrap-samples 1000 --bootstrap-confidence 0.90
uv run python scripts/generate_figures.py \
  --episodes output/benchmarks/episodes.jsonl \
  --table-summary output/benchmarks/summary_ci90.json \
  --table-metrics collisions,snqi \
  --table-stats mean,median \
  --table-include-ci --ci-column-suffix ci90 --table-tex \
  --no-pareto --out-dir docs/figures/ci90_example
````

Fast iteration tip:

* Use `--no-pareto` with `scripts/generate_figures.py` to skip Pareto plot during rapid table refinement.
* Restrict distributions via `--dmetrics collisions,snqi` for quick rebuilds.

### Per-Test Performance Budget

A performance budget for tests helps prevent runtime regressions:

* Soft threshold: <20s (advisory)
* Hard timeout: 60s (enforced via `@pytest.mark.timeout(60)` markers)
* Report: Top 10 slowest tests printed with guidance at session end
* Relax: `ROBOT_SF_PERF_RELAX=1` suppresses soft breach failure escalation
* Enforce: `ROBOT_SF_PERF_ENFORCE=1` converts any soft or hard breach into a failure (unless relax set); advanced internal overrides: `ROBOT_SF_PERF_SOFT`,    `ROBOT_SF_PERF_HARD`.

Core helpers live in `tests/perf_utils/` (policy, guidance, reporting, minimal_matrix). See the development guide section for authoring guidance and troubleshooting steps: [Dev Guide – Per-Test Performance Budget](./dev_guide.md#per-test-performance-budget).

### ⚙️ Setup & Configuration

* [**UV Migration**](./UV_MIGRATION.md) - Migration to UV package manager
* [**Subtree Migration**](./SUBTREE_MIGRATION.md) - Git subtree integration for fast-pysf (migration from submodule)

### 📈 Pedestrian Metrics

* [**Pedestrian Metrics Overview**](./ped_metrics/PED_METRICS.md) - Summary of implemented metrics and their purpose
* [**Metric Analysis**](./ped_metrics/PED_METRICS_ANALYSIS.md) - Overview of metrics used in research and validation
* [**NPC Pedestrian Design**](./ped_metrics/NPC_PEDESTRIAN.md) - Details on the design and behavior of NPC pedestrians

* [**Pedestrian Density Reference**](./ped_metrics/PEDESTRIAN_DENSITY.md) - Units, canonical triad (0.02/0.05/0.08), advisory range, difficulty mapping & test policy

* [Per-pedestrian force quantiles demo](../examples/benchmarks/per_ped_force_quantiles_demo.py) - Script comparing aggregated vs per-ped force quantiles
* [**Issue 503 Pedestrian-Impact Metrics Notes**](./context/issue_503_pedestrian_impact_metrics.md) - Execution notes for the optional experimental `ped_impact_*` metric group
* [**Issue 1085 Pedestrian-Impact Aggregate Metrics**](./context/issue_1085_pedestrian_impact_metrics.md) - Schema-backed `pedestrian-impact.v1` block, aggregate reduction path, and opt-in CLI contract

### 📁 Media Resources

* [`img/`](./img/) - Documentation images and diagrams
* [`video/`](./video/) - Demo videos and animations

## 🚀 Quick Start Guides

### New Environment Architecture (Recommended)

```python
# Modern factory pattern for clean environment creation
from robot_sf.gym_env.environment_factory import (
    make_robot_env,
    make_image_robot_env,
    make_pedestrian_env
)

# Create environments with consistent interface
robot_env = make_robot_env(debug=True)
image_env = make_image_robot_env(debug=True)
ped_env = make_pedestrian_env(robot_model=model, debug=True)
```

### Legacy Class Pattern

#### Environment Factory Ergonomics Migration (Feature 130)

See the migration guide: [Environment Factory Migration](./dev/issues/130-improve-environment-factory/migration.md). Includes before/after examples, seeding, explicit replacements for retired legacy kwargs, and precedence rules. Quickstart examples: `specs/130-improve-environment-factory/quickstart.md` .

```python
# Direct class construction remains available when needed
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.env_config import EnvSettings

env = RobotEnv(env_config=EnvSettings(), debug=True)
```

## 🎯 Key Features

### Environment System

* **Unified Architecture**: Consistent base classes for all environments
* **Factory Pattern**: Clean, intuitive environment creation
* **Configuration Hierarchy**: Structured, extensible configuration system
* **Backward Compatibility**: Existing code continues to work

### Simulation Capabilities

* **Multi-Agent Support**: Robot and pedestrian simulation
* **Advanced Sensors**: LiDAR, image observations, target sensors
* **Map Integration**: Support for SVG maps and OpenStreetMap data
* **Visualization**: Real-time rendering and video recording

### Training & Analysis

* **Gymnasium Integration**: Compatible with modern RL frameworks
* **StableBaselines3 Support**: Ready for SOTA RL algorithms
* **Data Analysis Tools**: Comprehensive analysis utilities
* **Performance Monitoring**: Built-in metrics and logging
* **Multi-Extractor Workflow**: `scripts/multi_extractor_training.py` writes timestamped runs under `tmp/multi_extractor_training/`, capturing JSON + Markdown summaries alongside legacy `complete_results.json` for downstream automation.

## 📖 Documentation Highlights

### 🆕 Latest Updates

* **Architecture Decoupling (Feature 149)**: Simulator facade and registries (simulator & sensors) scaffolded behind the factory pattern; backend selection via unified config with a default of "fast-pysf". See design docs and quickstart below.

### 📋 Migration Status

### Architecture & design features

* Architectural decoupling and consistency overhaul (Feature 149):
  + Design: `specs/149-architectural-coupling-and/spec.md`
  + Plan: `specs/149-architectural-coupling-and/plan.md`
  + Quickstart: `specs/149-architectural-coupling-and/quickstart.md`

* **33 files** identified for migration to new pattern
* **Migration script** available for automated updates
* **Full documentation** provided for smooth migration

## 🔗 External Links

* [**Project Repository**](https://github.com/ll7/robot_sf_ll7) - Main GitHub repository
* [**Gymnasium Documentation**](https://gymnasium.farama.org/) - RL environment framework
* [**StableBaselines3**](https://stable-baselines3.readthedocs.io/) - RL algorithms
* [**PySocialForce**](https://github.com/svenkreiss/PySocialForce) - Pedestrian simulation

## 🤝 Contributing

When contributing to the project:

1. **Use the new factory pattern** for environment creation
2. **Follow the unified configuration system** for settings
3. **Use the [planner contribution guide](./contributing_planner.md)** when adding planner or adapter surfaces
4. **Check the migration guide** when updating existing code
5. **Run tests** to ensure compatibility with both old and new patterns

## 📞 Support

* **Environment Issues**: Check the [refactoring documentation](./refactoring/)
* **Migration Help**: Use the [migration guide](./refactoring/migration_guide.md)
* **General Questions**: See individual documentation files
* **Bug Reports**: Use the GitHub issue tracker

## Planner Documentation

* **Planner contribution guide**: See `docs/contributing_planner.md` for the minimum planner/adapter contribution flow, metadata, config-first invocation, smoke proof, registry, and benchmark-boundary checklist.
* **Global Planner**: See `specs/342-svg-global-planner/quickstart.md` for the visibility-graph planner API, POI routing, and integration guidance.
* **Planner selection**: Choose between visibility and classic grid planners in `docs/dev_guide.md#planner-selection-visibility-vs-classic-grid`.
* **MPC social-navigation spike**: See `docs/context/issue_771_drmpscnav_assessment.md` and `docs/context/issue_771_drmpscnav_implementation_guide.md` for the SICNav / DR-MPC assessment boundary, implementation guardrails, and verified-simple gate plan.
* **ACMPC learned-MPC feasibility**: See `docs/context/issue_3985_acmpc_feasibility_assessment.md` for the assessment-only boundary for an Actor-Critic Model Predictive Control inspired local planner.

---

_Last updated: April 2026 - Paper-matrix extended seed schedule context added_
