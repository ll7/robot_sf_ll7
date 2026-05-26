# Issue #1530 Optional Planner Preflight Audit

## Goal

Audit benchmark-facing optional planner/runtime dependencies beyond the existing ORCA `rvo2`
check, then add only the fail-closed guards that current `main` needs to keep benchmark evidence
honest.

## Scope And Anchors

- Benchmark policy: `docs/context/issue_691_benchmark_fallback_policy.md`
- Wrapper entrypoints:
  `scripts/tools/run_camera_ready_benchmark.py`,
  `scripts/tools/run_benchmark_release.py`
- Benchmark preflight/runtime seam:
  `robot_sf/benchmark/map_runner.py`
- Planner families reviewed:
  `robot_sf/planner/social_navigation_pyenvs_*.py`,
  `robot_sf/planner/crowdnav_height.py`,
  `robot_sf/planner/sonic_crowdnav.py`,
  `robot_sf/planner/socnav.py`,
  `robot_sf/baselines/{ppo,sac,drl_vo,dr_mpc,sicnav}.py`

## Inventory And Classification

| Surface | Optional dependency / asset | Current behavior on current `main` | Benchmark classification |
| --- | --- | --- | --- |
| `orca`, `social_navigation_pyenvs_orca` | `rvo2` | Camera-ready CLI already preflighted; release CLI bypassed that guard | **Guard added** in release wrapper |
| `social_navigation_pyenvs_socialforce` | upstream checkout, `socialforce==0.2.3`, Torch-backed backend | Adapter constructor fails before episode work if checkout/runtime is missing or incompatible | Already fail-closed |
| `social_navigation_pyenvs_sfm_helbing` | upstream checkout | Adapter constructor fails before episode work if checkout/runtime is missing | Already fail-closed |
| `social_navigation_pyenvs_hsfm_new_guo` | upstream checkout | Adapter constructor fails before episode work if checkout/runtime is missing | Already fail-closed |
| `crowdnav_height` | upstream checkout + extracted checkpoint bundle + Torch | Adapter constructor fails before episode work if checkout/checkpoint is missing | Already fail-closed |
| `sonic_crowdnav`, `gensafenav_*` | upstream checkout + checkpoint + Torch | Adapter constructor fails before episode work if checkout/checkpoint/runtime is missing or incompatible | Already fail-closed |
| `socnav_sampling`, `sacadrl`, `prediction_planner`, `socnav_bench`, hybrid SocNav wrappers | SocNavBench checkout, TensorFlow, checkpoints, `skfmm`, model assets | `map_runner._preflight_policy()` already marks `skip`/`fallback` explicitly via `socnav_missing_prereq_policy` | Already explicit, non-silent |
| `sicnav`, `dr_mpc` | external upstream package / checkout | `_build_policy()` raises immediately when metadata reports missing dependency | Already fail-closed |
| `ppo`, `sac`, `drl_vo` | SB3 / Torch / checkpoint resolution | Planner constructors could boot into internal goal-seeking fallback and preflight still returned `ok` | **Guard added** in benchmark preflight |

## Changes Made

1. `scripts/tools/run_benchmark_release.py` now runs `check_orca_rvo2_preflight(cfg)` immediately
   after loading the canonical campaign config, matching the camera-ready CLI's existing
   fail-fast behavior while preserving the release CLI's structured JSON exit contract.
2. `robot_sf/benchmark/map_runner.py` now inspects planner metadata returned by `_build_policy()`
   during preflight and converts `status="fallback"` into `preflight.status="skipped"` before any
   episode jobs are scheduled. This closes the benchmark-strength evidence gap for learned
   planners such as PPO, SAC, and DRL-VO when their optional runtime/checkpoint dependencies are
   absent and they would otherwise silently goal-seek.

## Why No Broader Guard Was Added

Most other optional-dependency planners already fail early at adapter construction or are routed
through the explicit SocNav prereq policy. Adding another wrapper-level inventory pass for those
surfaces would duplicate constructor checks without changing benchmark semantics on current `main`.

## Validation Path

- `pytest tests/benchmark/test_map_runner_utils.py`
- `pytest tests/benchmark/test_map_runner_preflight_profiles.py`
- `pytest tests/tools/test_run_benchmark_release.py`
- `scripts/dev/ruff_fix_format.sh`
- `git diff --check origin/main...HEAD`
- `BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh`
