# Issue 649 Social-Navigation-PyEnvs ORCA Parity Note

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#642` Social-Navigation-PyEnvs source-harness reproduction
- `robot_sf_ll7#644` Prototype benchmark-facing Social-Navigation-PyEnvs ORCA integration
- `robot_sf_ll7#649` Validate source parity for Social-Navigation-PyEnvs ORCA integration

## Goal

Determine whether the poor paper-surface performance of `social_navigation_pyenvs_orca` is
primarily a benchmark/scenario mismatch or a material contract mismatch in the current Robot SF
wrapper.

## Method

Compare the current Robot SF adapter against the upstream
`crowd_nav.policy_no_train.orca.ORCA.predict()` path on fixed upstream scenarios.

For each traced decision point:

- run the upstream `SocialNavSim` with `crowdnav_policy=True`
- capture the upstream `ActionXY` chosen by the upstream robot policy
- reconstruct a Robot SF-style observation from the same upstream snapshot
- call the current `robot_sf/planner/social_navigation_pyenvs_orca.py` adapter
- record the adapter's raw upstream `ActionXY` before `unicycle_vw` projection
- run an oracle-heading control that replaces Robot SF yaw with the true velocity heading from the
  upstream simulator

This isolates the self-state mapping from downstream benchmark execution.

## Canonical validation command

```bash
uv run python scripts/tools/probe_social_navigation_pyenvs_orca_parity.py \
  --repo-root output/repos/Social-Navigation-PyEnvs \
  --output-json output/benchmarks/external/social_navigation_pyenvs_orca_parity_probe/report.json \
  --output-md output/benchmarks/external/social_navigation_pyenvs_orca_parity_probe/report.md
```

## Upstream reference

- upstream repo: <https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs>
- local checkout path: `output/repos/Social-Navigation-PyEnvs`
- upstream policy path: `crowd_nav.policy_no_train.orca.ORCA`
- compared simulator path: `social_gym.social_nav_sim.SocialNavSim` with
  `set_robot_policy(policy_name="orca", crowdnav_policy=True)`

## Current result

Observed artifact:

- `output/benchmarks/external/social_navigation_pyenvs_orca_parity_probe/report.md`

Observed verdict:

- `adapter has material contract mismatch`

Observed scenario summaries:

- `circular_crossing_hsfm_new_guo`
  - wrapper mean ActionXY error: `0.0000`
  - wrapper max ActionXY error: `0.0000`
  - oracle-heading mean ActionXY error: `0.0000`
  - heading/velocity mismatch steps: `0`
- `parallel_traffic_orca`
  - wrapper mean ActionXY error: `0.0754`
  - wrapper max ActionXY error: `0.2974`
  - oracle-heading mean ActionXY error: `0.0000`
  - heading/velocity mismatch steps: `6`

## Interpretation

What matched:

- on straight-line `circular_crossing_hsfm_new_guo`, upstream `ActionXY`, wrapper `ActionXY`, and
  oracle-heading `ActionXY` stayed aligned

What failed:

- on `parallel_traffic_orca`, the wrapper diverged from the upstream `ActionXY` as soon as the
  upstream robot velocity heading stopped matching the robot yaw
- the first clear divergence occurs at traced decision time `t=0.25`, where upstream ORCA chooses
  `[0.3618, -0.1754]` but the current wrapper produces `[0.5919, 0.0130]`
- the oracle-heading control restored near-exact parity on those same snapshots

Root-cause judgment:

- the current Robot SF wrapper reconstructs self velocity from `heading + scalar speed`
- the upstream ORCA path consumes full planar self velocity `(vx, vy)`
- that mismatch becomes material once upstream ORCA generates lateral motion

Projection judgment:

- `ActionXY -> unicycle_vw` projection is still a second fidelity risk
- but the probe shows raw upstream `ActionXY` parity already fails before projection

## Claim boundary

What this issue proves:

- the current poor paper-surface performance of `social_navigation_pyenvs_orca` is not just a vague
  “weak planner” result
- there is a concrete self-velocity contract mismatch in the current wrapper

What this issue does not prove:

- that the upstream ORCA policy itself is weak
- that a corrected parity wrapper would perform well on the Robot SF paper surface
- that the current benchmark-facing prototype should be promoted beyond `experimental`

## Recommendation

- keep `social_navigation_pyenvs_orca` as an upstream-backed prototype entry
- do not treat its current paper-surface numbers as source-faithful ORCA-family evidence
- if this planner family remains interesting, the next implementation step should correct or
  explicitly redesign the self-velocity contract before any stronger performance interpretation
