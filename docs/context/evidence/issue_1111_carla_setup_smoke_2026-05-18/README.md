# Issue #1111 CARLA Setup-Only Smoke Evidence

Issue: [#1111](https://github.com/ll7/robot_sf_ll7/issues/1111)
Parent epic: [#872](https://github.com/ll7/robot_sf_ll7/issues/872)
Runtime substrate: [#1179](https://github.com/ll7/robot_sf_ll7/issues/1179)

## Contents

This bundle preserves the small, reviewable artifacts from the May 18, 2026 setup-only CARLA T1
oracle smoke run:

- `carla_availability.json`: `robot-sf-check-carla --json --require` result using ephemeral
  `carla==0.9.16`.
- `manifest.json`: T0 export manifest generated from
  `configs/scenarios/single/pr_promoted_planner_smoke.yaml`.
- `pr_promoted_planner_smoke.json`: selected T0 payload.
- `setup_smoke.json`: setup-only T1 oracle smoke summary.
- `manifest.sha256`: checksums for the copied evidence files.

## Commands

```bash
rtk uv run robot-sf-export-carla-t0 \
  --scenario-file configs/scenarios/single/pr_promoted_planner_smoke.yaml \
  --output-dir output/carla_bridge/issue1111_setup_only_t0 \
  --robot-sf-commit 10c64e93 \
  --created-by goal-issue-implementation \
  --certificate-generator scenario_cert.v1

rtk uv run robot-sf-validate-carla-t0-manifest \
  --manifest output/carla_bridge/issue1111_setup_only_t0/manifest.json

rtk uv run robot-sf-validate-carla-t0-batch \
  --manifest output/carla_bridge/issue1111_setup_only_t0/manifest.json --json

rtk uv run --with carla==0.9.16 robot-sf-check-carla --json --require

rtk bash -lc 'uv run --with carla==0.9.16 robot-sf-carla-t1-oracle-smoke \
  --manifest output/carla_bridge/issue1111_setup_only_t0/manifest.json --json \
  > output/carla_bridge/issue1111_setup_only_t1_smoke.json'
```

## Boundary

This is setup-only evidence. It proves the optional CARLA Python API boundary and T0 payload
selection path with `carla==0.9.16`, but it does not start a CARLA server, spawn actors, replay a
world, compare metrics, or support simulator-transfer claims. Live replay remains Issue #1169.
