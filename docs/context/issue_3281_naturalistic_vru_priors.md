# Issue #3281 Naturalistic VRU Priors

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/3281>

Status: implementation contract for generated adversarial manifests. This note does not claim
real-world calibration, adversarial coverage, planner weakness, or benchmark-strength evidence.

## What Was Built

Generated `adversarial_scenario_manifest.v1` records now include an additive
`naturalistic_prior` block:

- `schema_version: naturalistic_vru_prior.v1`
- `profile: urban_vru_default_v1`
- inclusive interpretable bounds for `pedestrian_speed_mps`, `pedestrian_delay_s`, and
  `spawn_time_s`
- optional explicit-control bounds for `pedestrian_acceleration_mps2` and `group_size` when those
  controls are present in the generated manifest
- cyclist-specific speed and acceleration bounds when `candidate_controls.vru_profile: cyclist`
  selects the cyclist profile
- `passed` plus `violation_flags`
- per-constraint observed value and pass/fail metadata

The v1 prior intentionally uses only explicit generated controls in `CandidateSpec` and
`candidate_controls`. Acceleration, group-size, and cyclist checks are evaluated only when those
controls are present in the manifest. Social Force, IDM/MOBIL, trajectory-derived acceleration,
route curvature, and density priors remain deferred because the generated manifest surface does not
carry those values explicitly.

## Claim Boundary

`naturalistic_prior.passed: true` means the generated candidate is plausible under this authored
local bounds profile only. It is not a paper-facing realism claim.

`naturalistic_prior.passed: false` separates an intentionally unrealistic or stress-only generated
candidate from plausible hard cases. It does not automatically make the manifest structurally
invalid; downstream tooling can filter or report the candidate by naturalness status.

`candidate_controls.vru_profile` is a profile selector, not a numeric naturalistic constraint. The
generated-candidate schema accepts `pedestrian_acceleration_mps2` and `group_size` as numeric
constraint fields, while profile selection is represented by the prior `profile` value.

## Quality Summary

`adversarial_manifest_quality_summary.v1` now reports:

- naturalistic-prior availability, pass, fail, and missing counts;
- pass/fail rates among manifests that include prior metadata;
- violation counts by flag;
- `--naturalistic-status passed|violated|missing|all` filtering in
  `scripts/tools/summarize_adversarial_manifest_quality.py`.

Legacy manifests without `naturalistic_prior` are classified as `missing`, not failed.

## Validation Plan

Focused proof should include:

```bash
uv run python -m pytest \
  tests/adversarial/test_adversarial_scenario_manifest.py \
  tests/adversarial/test_adversarial_manifest_quality.py \
  tests/benchmark/test_generated_scenario_candidate_schema.py -q
uv run python scripts/tools/summarize_adversarial_manifest_quality.py --help
```
