# Issue 3266 PPO/SNQI Blocker Closeout

Plain-language summary: the immediate Proximal Policy Optimization (PPO) and Social Navigation
quality index (SNQI) blockers are repaired for the smallest valid scenario-horizon smoke slice, but
this is not paper-facing Results evidence.

- Related issue: <https://github.com/ll7/robot_sf_ll7/issues/3266>
- Repair PRs: <https://github.com/ll7/robot_sf_ll7/pull/3442>,
  <https://github.com/ll7/robot_sf_ll7/pull/3443>
- Evidence bundle:
  `docs/context/evidence/issue_3266_ppo_snqi_smoke_2026-06-23/`
- Source config: `configs/benchmarks/issue_3266_scenario_horizon_ppo_snqi_smoke.yaml`
- Evidence status: `valid_blocker_resolution_smoke`

## Claim Boundary

The issue #3266 blocker-resolution slice is valid smoke evidence: PPO runs natively on the
scenario-horizon smoke path, the learned-policy contract passes, the SNQI contract passes, and the
tracked summary reports zero fallback, degraded, or unexpected failed rows.

This closeout does not promote PPO/SNQI to paper-facing scenario-horizon Results evidence. A broader
paper-facing scenario-horizon campaign still needs its own predeclared config, run, artifact
provenance, and evidence packet before any dissertation or Results wording cites PPO/SNQI as a full
scenario-horizon result.

## Status

The immediate validity blocker should be treated as resolved at smoke tier. Any future work on
paper-facing promotion should start from the tracked smoke config and evidence bundle above, keep
fallback/degraded rows excluded from success evidence, and open a new bounded issue or PR for the
broader Results promotion step.
