# Issue 656 Social-Navigation-PyEnvs SocialForce Retry Note

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#646` Prototype benchmark-facing Social-Navigation-PyEnvs SocialForce/SFM integration
- `robot_sf_ll7#653` Prototype compatible Social-Navigation-PyEnvs SocialForce runtime reproduction
- `robot_sf_ll7#656` Prototype benchmark-facing Social-Navigation-PyEnvs SocialForce retry with explicit runtime shim

## Goal

Retry the benchmark-facing `social_navigation_pyenvs_socialforce` path now that the runtime blocker
has been removed with an explicit compatibility shim for `socialforce==0.2.3`.

## Scope

- keep the upstream `crowd_nav.policy_no_train.socialforce.SocialForce` policy logic unchanged,
- make the compatibility shim explicit in the Robot SF adapter and metadata,
- prove end-to-end execution on the sanity surface,
- then run the canonical paper surface and classify the planner conservatively.

## Result

Verdict: `runtime corrected, but still weak and partially unstable`

Canonical paper-surface campaign:

- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_social_navigation_pyenvs_socialforce_only_issue656_social_navigation_pyenvs_socialforce_paper_surface_20260320_154139`

Observed result:

- status: `partial-failure`
- jobs written: `83 / 141`
- success: `0.0000`
- collisions: `0.0241`
- runtime: `101.0591s`
- SNQI: `nan`
- projection rate: `0.0000`
- infeasible rate: `0.0000`
- failed jobs: `58`
- dominant failure: `ValueError('cannot convert float NaN to integer')`

## Interpretation

What improved:

1. the old constructor mismatch is gone,
2. the upstream policy now runs through an explicit compatibility shim,
3. the benchmark metadata makes that shim boundary explicit.

What did not improve enough:

1. the planner is not competitive on the paper surface,
2. the run is still unstable across many scenarios,
3. the current failure mode is now NaN propagation rather than missing runtime support.

This is therefore not a benchmark-quality planner addition. It remains useful as integration
evidence, but not as a headline baseline.

## Comparison boundary

Compared against the stronger currently tracked baselines:

- frozen canonical `orca`: success `0.2340`, collisions `0.0426`
- `social_navigation_pyenvs_orca`: success `0.0213`, collisions `0.0638`
- `social_navigation_pyenvs_sfm_helbing`: success `0.0142`, collisions `0.0922`

`social_navigation_pyenvs_socialforce` does not outperform any of these in a way that justifies
promotion. It also fails to complete the full paper surface cleanly.

## Final verdict

- `social_navigation_pyenvs_socialforce`: `integration proof only`
- current status: `not promising as a benchmark-quality upgrade`
