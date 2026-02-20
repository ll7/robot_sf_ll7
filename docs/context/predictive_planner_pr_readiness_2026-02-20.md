# Predictive Planner PR Readiness (2026-02-20)

This checklist captures what is done and what still needs explicit maintainer confirmation before opening a final PR.

## Completed

- Predictive planner integrated as benchmark algorithm (`prediction_planner`).
- Model ids registered in `model/registry.yaml`.
- Default predictive model id switched to `predictive_proxy_selected_v1`.
- Camera-ready benchmark presets include `prediction_planner` with explicit algo config:
  - `configs/algos/prediction_planner_camera_ready.yaml`
  - `configs/benchmarks/camera_ready_all_planners.yaml`
  - `configs/benchmarks/camera_ready_smoke_all_planners.yaml`
  - `configs/benchmarks/camera_ready_all_planners_strict_socnav.yaml`
- High-level baseline documentation added:
  - `docs/baselines/prediction_planner.md`
- Training process documented:
  - `docs/training/predictive_planner_training.md`
- Script catalog updated for predictive workflow:
  - `scripts/README.md`

## Needs Maintainer Decision (before final PR merge)

1. Checkpoint distribution policy:
   - Keep local-only checkpoint paths (current), or publish portable checkpoint artifacts and add registry download metadata.
2. Experimental scope in camera-ready campaigns:
   - Keep `prediction_planner` enabled by default in all-planners presets (current), or gate behind a separate config.
3. Promotion criteria:
   - Define hard-seed success threshold required to move from `experimental` to stronger benchmark claim.

## Suggested PR Description Bullets

- Introduces prediction planner integration and risk-aware adaptive scoring/lattice.
- Provides full training/evaluation runbook and baseline documentation with citation provenance.
- Adds camera-ready benchmark config support for `prediction_planner`.
- Leaves planner tier as `experimental` pending hard-case success improvements and artifact distribution decisions.
