# Issue #1432 Adaptive Test Strategy Claim Audit (2026-05-22)

> **Scope:** Docs-only claim hygiene for GitHub Issue #1432.
> **Date:** 2026-05-22
> **Target path:** `docs/context/adaptive_test_claim_audit_2026-05.md`
> **Sensitivity:** paper-facing claim hygiene. No code, test, workflow, schema, config, metric,
> formula, benchmark behavior, or paper text changes.

## Goal

Inventory the repository surfaces that relate to "adaptive testing," "adaptive test strategy," or similar claims, classify what actually executes versus what is proposed or absent, and render a conservative verdict on whether the repository currently supports a strong "adaptive test strategy" claim.

## Method Inventory

| Method | Repo surface | Evidence link | Execution status | Claim strength |
|--------|-------------|---------------|------------------|----------------|
| Fixed-budget adversarial random search | `robot_sf/adversarial/search.py`, `SearchConfig`, `run_adversarial_search` | [issue_869_adversarial_runner.md](issue_869_adversarial_runner.md) | **Executed** — deterministic sequential search with configurable budget, seed, and horizon; replay bundles emitted under `output/` | Strong for "bounded adversarial search"; weak for "adaptive" because budget is fixed before execution and no online stopping rule exists |
| Coordinate-refinement sampler | `robot_sf/adversarial/samplers.py`, `CoordinateRefinementSampler` | [issue_869_adversarial_runner.md](issue_869_adversarial_runner.md) | **Executed** — deterministic bounded coordinate perturbations around best observed candidate | Strong for "bounded local refinement"; weak for "adaptive" because it lacks dynamic reallocation of budget or termination based on convergence |
| Optuna-backed adversarial sampler | `robot_sf/adversarial/samplers.py`, `OptunaCandidateSampler` | [issue_1236_optimizer_adversarial_sampler.md](issue_1236_optimizer_adversarial_sampler.md) | **Executed** — pilot only; synthetic comparison at budget 3 showed no objective improvement over random; explicitly marked non-paper-facing | Weak / pilot-only; not benchmark evidence |
| Adversarial failure archive | `robot_sf/adversarial/archive.py`, `curate_adversarial_failure_archive.py` | [issue_1237_adversarial_failure_archive.md](issue_1237_adversarial_failure_archive.md) | **Executed** — deterministic grouping and representative selection from search manifests; replay command pointers preserved | Strong for "failure curation"; does not constitute adaptive test execution |
| Seed-sensitivity explorer | `robot_sf/adversarial/seed_sensitivity.py`, `run_seed_sensitivity` | [issue_1271_seed_sensitivity_explorer.md](issue_1271_seed_sensitivity_explorer.md) | **Executed** — replays one fixed candidate across explicit seed grids; classifies `stable_failure` vs `brittle_failure` | Strong for "seed-sensitivity classification"; not dynamic adaptive search |
| Timing/speed perturbation grid | `robot_sf/adversarial/seed_sensitivity.py` perturbation helpers | [issue_1294_seed_sensitivity_perturbations.md](issue_1294_seed_sensitivity_perturbations.md) | **Executed** — bounded timing/speed perturbations around adversarial candidates with fail-closed rejection of unbounded deltas | Strong for "bounded perturbation replay"; not adaptive stopping |
| Scenario coverage entropy report | `robot_sf/benchmark/scenario_coverage.py`, `scenario_coverage_entropy.py` | [issue_1240_scenario_coverage_entropy.md](issue_1240_scenario_coverage_entropy.md) | **Executed** — config-only diagnostic report using authored scenario metadata; explicitly not benchmark-success or safety evidence | Strong for "diagnostic coverage accounting"; explicitly non-adaptive and non-executional |
| SNQI weight optimization (grid + differential evolution) | `scripts/snqi_weight_optimization.py`, unified CLI `robot_sf_bench snqi optimize` | [docs/snqi-weight-tools/README.md](../snqi-weight-tools/README.md), [issue_1286_snqi_bootstrap_stability.md](issue_1286_snqi_bootstrap_stability.md) | **Executed** — deterministic grid search and SciPy differential evolution for weight-space search; bootstrap stability added | Strong for "weight-space search"; not scenario-level adaptive testing |
| Optuna feature-extractor sweep | `scripts/training/optuna_feature_extractor.py`, SLURM array launcher | [issue_193_feature_extractor_optuna_study.md](issue_193_feature_extractor_optuna_study.md) | **Executed** — 4 M-step SLURM array sweep (38 completed trials); 12 M hardening batch evaluated; hold-out policy analysis rejected promotion (0.727 success, 0.273 collision, well below gate) | Strong for "hyperparameter sweep execution"; weak for "adaptive test" because the test surface (benchmark evaluation) is fixed and the adaptation is only at training-time architecture selection |
| Optuna expert PPO sweep | `scripts/training/optuna_expert_ppo.py` | [docs/training/optuna_expert_ppo_sweep_2026-02-11.md](../training/optuna_expert_ppo_sweep_2026-02-11.md) | **Executed** — hyperparameter sweep; sweep report exists with recommended ranges | Strong for "training hyperparameter sweep"; not adaptive testing of planner behavior |
| Probabilistic / MCTS-lite predictive planner | `prediction_planner_probabilistic_cvar.yaml`, `prediction_planner_mcts_lite.yaml` | [issue_591_prediction_planner_probabilistic_search.md](issue_591_prediction_planner_probabilistic_search.md) | **Executed** — paper-surface benchmark run completed; CVaR and MCTS-lite are 4.1x and 5.1x slower, improve collision slightly, lose success, remain **experimental only** | Weak; explicitly rejected for promotion; not an adaptive test method |
| Scenario-adaptive hybrid ORCA candidate | `configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v1.yaml` | [issue_1023_experimental_benchmark_candidates.md](issue_1023_experimental_benchmark_candidates.md) | **Executed** — scenario-explicit algorithm selector (hybrid rule for most scenarios, tuned ORCA for leave-group); 144 episodes, success 0.9097, collision 0.0278 | Strong for "scenario-configured selector"; this is static config dispatch, not dynamic adaptive testing during execution |
| Corridor-deadlock evaluation slice | `configs/scenarios/sets/issue_1318_teb_corridor_deadlock_slice.yaml` | [issue_1318_teb_corridor_deadlock_eval.md](issue_1318_teb_corridor_deadlock_eval.md), [issue_1361_motion_primitive_corridor_deadlock.md](issue_1361_motion_primitive_corridor_deadlock.md) | **Executed** — 5-episode narrow slice; TEB 0/5 success, ORCA 2/5 success, hybrid-rule 4/5 success with 1 pedestrian collision | Strong for "narrow corridor-deadlock evidence"; not an adaptive test strategy |

### Bayesian / Optuna Tooling Classification

| Tooling | Classification | Rationale |
|---------|---------------|-----------|
| Optuna (TPE sampler) for adversarial candidate generation | **Optional-dependency-only pilot** | `optuna` is in `pyproject.toml`; `OptunaCandidateSampler` exists and executes, but the pilot evidence is synthetic/budget-3 and explicitly non-paper-facing. No real benchmark evaluation with adaptive stopping has been shown. |
| Bayesian optimization (general concept) | **Proposed / absent** | Specs mention Bayesian optimization as preferable to grid search ([specs/1024-retrain-learned-planners-h500/prompt_1024-retrain-learned-planner-h500.md](../../specs/1024-retrain-learned-planners-h500/prompt_1024-retrain-learned-planner-h500.md)), but no Bayesian optimizer (e.g., `scikit-optimize`, BoTorch) is integrated into the adversarial search or benchmark runner. |
| Optuna for feature-extractor / PPO hyperparameter search | **Used** | `optuna_feature_extractor.py` and `optuna_expert_ppo.py` are real executed surfaces with SLURM array evidence. |
| SNQI weight optimization (grid + differential evolution) | **Used** | `scripts/snqi_weight_optimization.py` executes; differential evolution is a population-based global optimizer, not Bayesian, but it is a real executed search surface. |
| CMA-ES or Bayesian optimizer adapters for adversarial search | **Proposed / absent** | [issue_869_adversarial_runner.md](issue_869_adversarial_runner.md) explicitly defers CMA-ES or Bayesian optimization to future optional-dependency boundaries after scenario semantics are stable. |

## Verdict: "Adaptive Test Strategy"

**Rejected as a strong claim.** The repository currently supports **scenario-based stress testing** and **bounded adversarial/scenario exploration**, but it does not support a strong "adaptive test strategy" claim because no executed surface demonstrates dynamic search with adaptive stopping or online feedback-driven reallocation of evaluation budget during benchmark execution. What exists are fixed-budget search pilots (random, coordinate, Optuna sampler at budget 3), deterministic seed-sensitivity replay grids, and diagnostic coverage entropy reports. The adversarial search API accepts feedback-capable samplers via `observe(evaluation)`, yet the executed evidence stops at small synthetic comparisons and explicitly defers candidate-level parallelism and convergence-based termination. The scenario-adaptive hybrid ORCA candidate ([issue_1023_experimental_benchmark_candidates.md](issue_1023_experimental_benchmark_candidates.md)) is a static scenario-to-algorithm dispatch table, not an adaptive online strategy. MCTS-lite probabilistic search ([issue_591_prediction_planner_probabilistic_search.md](issue_591_prediction_planner_probabilistic_search.md)) is 5x slower and loses success relative to the deterministic baseline, remaining explicitly experimental. Until a surface proves that the test loop can stop early, reallocate candidates, or modify scenario parameters based on intermediate benchmark outcomes, the claim should remain "bounded adversarial exploration and scenario stress testing" rather than "adaptive test strategy."

## Concrete Uncovered Failure Mode (with Evidence)

**Failure mode:** Hold-out policy analysis for the best feature-extractor candidate (`dyn_large_med`) reveals a persistent obstacle-collision regression that was not caught by training return alone.

**Evidence:** The 12 M-step hardening batch selected `dynamics / large / [128,128] / dropout=0.0` as the best family ([issue_193_feature_extractor_optuna_study.md](issue_193_feature_extractor_optuna_study.md)). Its hold-out policy analysis (Slurm 12106) reached only **0.727 success** and **0.273 collision**, well below the promotion gate (success >= 0.85, collision <= 0.08). The collision breakdown showed 12 obstacle/wall collisions and 6 pedestrian collisions across 66 episodes. Repeated obstacle-collision hotspots included `classic_bottleneck_high` (3/3 seeds), `classic_merging_low` (3/3 seeds), and `classic_merging_medium` (3/3 seeds). This demonstrates that **training-time hyperparameter search (Optuna sweep) does not automatically validate benchmark safety**; a separate fixed benchmark evaluation is required, and the current adaptive-search surfaces do not close this gap. The failure was uncovered by standard fixed-benchmark evaluation, not by any adaptive test loop.

**Secondary concrete failure mode (corridor-deadlock):** On the Issue #1318 narrow
corridor-deadlock slice, the in-repo TEB planner collided on all five evaluated seeds
(`0/5` success), ORCA produced two successes with three pedestrian collisions, and even the
best-available command-lattice proxy
([issue_1361_motion_primitive_corridor_deadlock.md](issue_1361_motion_primitive_corridor_deadlock.md))
still recorded one pedestrian collision on `classic_merging_medium` seed `112`. This is a known,
uncovered, and documented failure mode with executable replay evidence.

## Absent / Proposed Surfaces

| Surface | Status | Source |
|---------|--------|--------|
| Dynamic adaptive stopping during benchmark execution | **Absent** | No issue or code implements online termination based on intermediate failure rates or convergence proxies during a benchmark campaign. |
| Issue #1433 / Issue #1434 | **Open / PR-backed, not merged evidence** | GitHub Issue #1433 and Issue #1434 exist as current design/schema work, with draft or open PRs outside `origin/main` as of this audit. Treat them as pending design inputs, not executed evidence, until the related PRs merge and their docs become part of the repository context stack. |
| Real-time scenario-parameter adaptation based on live metric feedback | **Proposed only** | [issue_1272_validation_falsification_strategy.md](issue_1272_validation_falsification_strategy.md) frames optimizer-backed falsification as a near-term lane, but the executed #1236 pilot is explicitly bounded and non-paper-facing. |

## Related Surfaces

- Issue #869 adversarial runner API and v1 boundary: [issue_869_adversarial_runner.md](issue_869_adversarial_runner.md)
- Issue #1236 Optuna adversarial sampler pilot: [issue_1236_optimizer_adversarial_sampler.md](issue_1236_optimizer_adversarial_sampler.md)
- Issue #1237 adversarial failure archive: [issue_1237_adversarial_failure_archive.md](issue_1237_adversarial_failure_archive.md)
- Issue #1271 seed-sensitivity explorer: [issue_1271_seed_sensitivity_explorer.md](issue_1271_seed_sensitivity_explorer.md)
- Issue #1294 seed-sensitivity perturbations: [issue_1294_seed_sensitivity_perturbations.md](issue_1294_seed_sensitivity_perturbations.md)
- Issue #1240 scenario coverage entropy: [issue_1240_scenario_coverage_entropy.md](issue_1240_scenario_coverage_entropy.md)
- Issue #591 MCTS-lite / probabilistic predictive planner: [issue_591_prediction_planner_probabilistic_search.md](issue_591_prediction_planner_probabilistic_search.md)
- Issue #193 feature-extractor Optuna study and 12 M hardening: [issue_193_feature_extractor_optuna_study.md](issue_193_feature_extractor_optuna_study.md)
- Issue #1318 TEB corridor-deadlock evaluation: [issue_1318_teb_corridor_deadlock_eval.md](issue_1318_teb_corridor_deadlock_eval.md)
- Issue #1361 command-lattice corridor-deadlock proxy: [issue_1361_motion_primitive_corridor_deadlock.md](issue_1361_motion_primitive_corridor_deadlock.md)
- Issue #1023 experimental scenario-adaptive candidates: [issue_1023_experimental_benchmark_candidates.md](issue_1023_experimental_benchmark_candidates.md)
- Validation/falsification roadmap: [issue_1272_validation_falsification_strategy.md](issue_1272_validation_falsification_strategy.md)
- SNQI weight tooling (grid + differential evolution): [docs/snqi-weight-tools/README.md](../snqi-weight-tools/README.md)
- Optuna expert PPO sweep report: [docs/training/optuna_expert_ppo_sweep_2026-02-11.md](../training/optuna_expert_ppo_sweep_2026-02-11.md)
- Benchmark fallback policy (fail-closed rule): [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md)

## Validation

This note is documentation-only. Expected validation for the owning branch:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```

No code, test, or config changes were made for this audit.
