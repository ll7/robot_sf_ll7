# Issue #1509 CARLA Native Fixture Evidence

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1509>

This bundle records the 2026-05-31 CARLA fixture certification attempt on
`auxme-imech036`.

## Files

- `scenario_cert_pr_promoted.json`: `scenario_cert.v1.batch` output for
  `configs/scenarios/single/pr_promoted_planner_smoke.yaml`.
- `preflight.json`: CARLA Docker/runtime prerequisite report.
- `t0/manifest.json`: exported CARLA T0 manifest.
- `t0/pr_promoted_planner_smoke.json`: exported T0 payload.
- `live_replay_pr_promoted.json`: live CARLA Docker replay output.
- `parity_report_pr_promoted.json`: conservative Robot-SF/CARLA parity adapter output.
- `SHA256SUMS`: checksums for the evidence files.

## Result

The scenario certificate is valid and exportable, and CARLA live replay reached `oracle-replay`.
However, the robot spawn was projected by `18.190854674447806 m`, so the replay is
`oracle-replay-adapted` / `coordinate_alignment.replay_mode=adapted`.

The parity report is therefore `unavailable`. This is a fail-closed certification result, not a
native/aligned parity success.
