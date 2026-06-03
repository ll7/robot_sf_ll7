# Issue #2154 AMMV-Aware Social Force Model Slice

Issue: [#2154](https://github.com/ll7/robot_sf_ll7/issues/2154)

## Scope

This note records the first executable AMMV-aware Social Force model slice. The branch adds an
optional speed-scaled AMMV repulsion term to `robot_sf/baselines/social_force.py`, a diagnostic
configuration at `configs/baselines/social_force_ammv_aware.yaml`, and targeted unit coverage in
`tests/baselines/test_social_force.py`.

## Evidence

- `uv run pytest tests/baselines/test_social_force.py -q`: passed.
- `uv run pytest tests -k 'social_force or ammv' -q`: passed.
- `uv run ruff check robot_sf/baselines/social_force.py tests/baselines/test_social_force.py`:
  passed.
- `uv run ruff format --check robot_sf/baselines/social_force.py tests/baselines/test_social_force.py`:
  passed.
- `uv run python -m py_compile robot_sf/baselines/social_force.py tests/baselines/test_social_force.py`:
  passed.

## Claim Boundary

Classification: `diagnostic`.

The AMMV term is disabled by default and covered by tests showing default compatibility,
diagnostics-enabled metadata, AMMV force activation, speed scaling, and the no-pedestrian zero-force
case. This is mechanism evidence only. It is not benchmark-strength, calibrated, or paper-facing
evidence, and it does not show improved planner performance on the research-v1 AMMV matrix.

## Next Proof Step

Run a paired classical Social Force versus AMMV-aware Social Force diagnostic on a named AMMV
scenario/config/seed and promote only a compact summary with command, config, seed, commit, and
fallback/degraded exclusions.
