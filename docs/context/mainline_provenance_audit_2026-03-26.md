# Mainline Provenance Audit: Social-Navigation and gym-collision Artifacts

Date: 2026-03-26

## Scope

This note records a re-check of `origin/main` for the artifacts that were initially suspected to be
missing from mainline:

- `docs/context/issue_641_gym_collision_avoidance_side_env.md`
- `docs/context/issue_653_social_navigation_pyenvs_socialforce_runtime.md`
- `docs/context/issue_659_gym_collision_avoidance_headless.md`
- `docs/context/issue_646_social_navigation_pyenvs_force_models_integration.md`
- `docs/context/issue_656_social_navigation_pyenvs_socialforce_retry.md`
- `robot_sf/benchmark/algorithm_metadata.py`
- `robot_sf/planner/social_navigation_pyenvs_force_model.py`
- `configs/benchmarks/social_navigation_pyenvs_force_models_probe.yaml`
- `scripts/tools/probe_gym_collision_avoidance_side_env.py`
- `scripts/tools/probe_gym_collision_avoidance_headless_reproduction.py`
- `scripts/tools/probe_social_navigation_pyenvs_socialforce_runtime.py`
- the matching `tests/tools/test_probe_*.py` files

## Result

`origin/main` already contains the above provenance, probe, and adapter surfaces.

That means the earlier branch-by-branch read was useful as guidance, but it overstated what main was
missing. The right interpretation is:

- no runtime or benchmark-facing backport is required from those branches,
- the existing issue/context docs are the right place to preserve the reasoning,
- and any future claim that a file is missing from main should be verified directly against
  `origin/main` before planning a cherry-pick.

## Guardrails

Before treating a branch-local artifact as missing, verify all three:

1. the file is absent from `origin/main`,
2. it adds reproducibility or benchmark provenance value,
3. it does not duplicate an existing mainline probe or context note.

If those checks fail, leave main unchanged and prefer documentation or issue commentary over code
movement.
