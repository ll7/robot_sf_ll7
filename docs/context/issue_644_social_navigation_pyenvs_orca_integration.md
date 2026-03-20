# Issue 644 Social-Navigation-PyEnvs ORCA Integration Note

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#642` Social-Navigation-PyEnvs source-harness reproduction
- `robot_sf_ll7#644` Prototype benchmark-facing Social-Navigation-PyEnvs ORCA integration

## Goal

Turn the `#642` wrapper proof into a benchmark-facing prototype planner entry without overstating
what has been proven.

## Upstream reference

- upstream repo: <https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs>
- local checkout path: `output/repos/Social-Navigation-PyEnvs`
- upstream policy path: `crowd_nav.policy_no_train.orca.ORCA`

## Adapter boundary

The integration remains thin and attributable to upstream ORCA.

Boundary:

- Robot SF provides a SocNav structured observation
- `robot_sf/planner/social_navigation_pyenvs_orca.py` maps that observation into the upstream
  `JointState` contract
- upstream `crowd_nav.policy_no_train.orca.ORCA.predict()` selects `ActionXY`
- Robot SF projects that upstream `ActionXY` into `unicycle_vw`

Benchmark-visible contract:

- upstream command space: `velocity_vector_xy`
- benchmark command space: `unicycle_vw`
- projection policy: `heading_safe_velocity_to_unicycle_vw`

## Canonical validation commands

Run the wrapper probe:

```bash
uv run python scripts/tools/probe_social_navigation_pyenvs_orca_wrapper.py \
  --repo-root output/repos/Social-Navigation-PyEnvs \
  --output-json output/benchmarks/external/social_navigation_pyenvs_orca_wrapper_probe/report.json \
  --output-md output/benchmarks/external/social_navigation_pyenvs_orca_wrapper_probe/report.md
```

Run the benchmark-facing prototype campaign:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/social_navigation_pyenvs_orca_probe.yaml \
  --label issue644_social_navigation_pyenvs_orca \
  --log-level WARNING
```

## Current result

Observed probe verdict:

- `output/benchmarks/external/social_navigation_pyenvs_orca_wrapper_probe/report.md`
- verdict: `wrapper prototype viable`

Observed benchmark run:

- campaign root:
  `output/benchmarks/camera_ready/social_navigation_pyenvs_orca_probe_issue644_social_navigation_pyenvs_orca_20260320_111051`
- report:
  `output/benchmarks/camera_ready/social_navigation_pyenvs_orca_probe_issue644_social_navigation_pyenvs_orca_20260320_111051/reports/campaign_report.md`

Key benchmark result:

- planner key: `social_navigation_pyenvs_orca`
- episodes: `3`
- success: `1.0000`
- collisions: `0.0000`
- execution mode: `adapter`
- readiness tier: `experimental`
- projection rate: `0.0000`
- infeasible rate: `0.0000`

Interpretation:

- the benchmark-facing planner entry runs successfully inside `robot_sf_ll7`
- provenance and projection semantics are now explicit in benchmark metadata
- this probe campaign is too small to support paper-facing quality claims

## Claim boundary

What this issue proves:

- `robot_sf_ll7` can execute a benchmark-facing prototype wrapper around upstream
  `Social-Navigation-PyEnvs` ORCA
- the upstream source, policy path, and projection semantics are explicit in benchmark metadata

What this issue does not prove:

- that this planner is better than canonical ORCA on the paper matrix
- that learned `Social-Navigation-PyEnvs` policies are ready for reuse
- that this prototype should replace the frozen paper ORCA entry

Current recommendation:

- keep this as an `experimental` planner-zoo prototype
- do not substitute it into paper-facing headline tables without a same-surface comparison first
