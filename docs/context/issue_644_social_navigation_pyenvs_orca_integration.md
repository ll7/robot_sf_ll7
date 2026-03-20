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

Run the same planner on the canonical paper surface:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_social_navigation_pyenvs_orca_only.yaml \
  --label issue644_social_navigation_pyenvs_orca_paper_surface \
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

## Same-surface paper comparison

Observed paper-surface run:

- campaign root:
  `output/benchmarks/camera_ready/paper_experiment_matrix_v1_social_navigation_pyenvs_orca_only_issue644_social_navigation_pyenvs_orca_paper_surface_20260320_112918`
- report:
  `output/benchmarks/camera_ready/paper_experiment_matrix_v1_social_navigation_pyenvs_orca_only_issue644_social_navigation_pyenvs_orca_paper_surface_20260320_112918/reports/campaign_report.md`

Comparison target:

- frozen canonical ORCA row from
  `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407/reports/campaign_summary.json`

Key same-surface metrics:

| Metric | Frozen `orca` | `social_navigation_pyenvs_orca` | Delta |
|---|---:|---:|---:|
| episodes | 141 | 141 | 0 |
| success | 0.2340 | 0.0213 | -0.2127 |
| collisions | 0.0426 | 0.0922 | +0.0496 |
| runtime_sec | 98.2864 | 31.0010 | -67.2854 |
| near_misses | 4.6950 | 4.3333 | -0.3617 |
| time_to_goal_norm | 0.9429 | 0.9979 | +0.0550 |
| comfort_exposure | 0.0315 | 0.0323 | +0.0008 |
| jerk | 0.1477 | 0.0502 | -0.0975 |
| SNQI | -0.2325 | -0.2907 | -0.0582 |
| projection_rate | 0.8195 | 0.0000 | -0.8195 |
| infeasible_rate | 0.8195 | 0.0000 | -0.8195 |

Interpretation:

- this upstream-backed ORCA path is much faster and does not hit the current ORCA projection /
  infeasibility behavior
- but it is materially worse on the benchmark outcomes that matter most here: success drops sharply,
  collisions increase, and overall SNQI degrades
- therefore this is an integration-proof and planner-zoo prototype, not a benchmark-quality upgrade

## Claim boundary

Superseded note:

- Issue `#651` corrected the adapter self-velocity contract and reran both parity and the paper
  surface.
- Treat the paper-surface metrics recorded below as the pre-fix prototype result, not the latest
  contract-corrected result.

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
- do not replace the frozen paper ORCA row with this variant
- use it as evidence that upstream-backed `Social-Navigation-PyEnvs` integration is feasible for
  non-trainable planners, not as evidence that this ORCA variant is benchmark-strong
