# Issue #1111 CARLA Setup-Only Smoke

Issue: [#1111](https://github.com/ll7/robot_sf_ll7/issues/1111)
Parent epic: [#872](https://github.com/ll7/robot_sf_ll7/issues/872)
Runtime substrate predecessor: [#1179](https://github.com/ll7/robot_sf_ll7/issues/1179)
Live replay successor: [#1169](https://github.com/ll7/robot_sf_ll7/issues/1169)
Evidence bundle:
[`docs/context/evidence/issue_1111_carla_setup_smoke_2026-05-18/`](evidence/issue_1111_carla_setup_smoke_2026-05-18/)

## Outcome

On May 18, 2026, after PR #1310 merged the pinned CARLA Docker runtime substrate, the setup-only
T1 oracle smoke path was run successfully with an ephemeral `carla==0.9.16` dependency:

```bash
rtk uv run --with carla==0.9.16 robot-sf-check-carla --json --require
```

Result: `status: available`, `dependency: carla`.

The selected T0 payload was exported from:

```text
configs/scenarios/single/pr_promoted_planner_smoke.yaml
```

The setup-only T1 smoke then produced `status: oracle-replay` and `stage: setup-only` for scenario
`pr_promoted_planner_smoke`.

## Boundary

This clears the Issue #1111 setup-only Python API and T0 payload-selection proof. It does not clear
the Issue #1169 live replay requirement because no CARLA server was started, no actors were spawned,
no world was replayed, and no Robot-SF/CARLA metric parity was attempted.

At the time of the May 18, 2026 setup-only smoke, the Docker-backed live runtime path from
Issue #1179 still failed closed on this machine with `missing_capability: docker-daemon`; Issue #1169
therefore remained blocked until a Docker/NVIDIA-capable host could run the Docker smoke or another
CARLA server path could be connected. That host-level blocker was superseded on May 31, 2026 for
`auxme-imech036`: see
[`issue_1179_carla_docker_runtime.md`](issue_1179_carla_docker_runtime.md) for the current local
CARLA Docker smoke command. This does not change the setup-only boundary of Issue #1111.

## Validation

```bash
rtk uv run robot-sf-export-carla-t0 \
  --scenario-file configs/scenarios/single/pr_promoted_planner_smoke.yaml \
  --output-dir output/carla_bridge/issue1111_setup_only_t0 \
  --robot-sf-commit 10c64e93 \
  --created-by goal-issue-implementation \
  --certificate-generator scenario_cert.v1
```

Result: wrote `output/carla_bridge/issue1111_setup_only_t0/manifest.json`.

```bash
rtk uv run robot-sf-validate-carla-t0-manifest \
  --manifest output/carla_bridge/issue1111_setup_only_t0/manifest.json
```

Result: `1 export`.

```bash
rtk uv run robot-sf-validate-carla-t0-batch \
  --manifest output/carla_bridge/issue1111_setup_only_t0/manifest.json --json
```

Result: one payload, `scenario_ids: ["pr_promoted_planner_smoke"]`.

```bash
rtk bash -lc 'uv run --with carla==0.9.16 robot-sf-carla-t1-oracle-smoke \
  --manifest output/carla_bridge/issue1111_setup_only_t0/manifest.json --json \
  > output/carla_bridge/issue1111_setup_only_t1_smoke.json'
```

Result: copied to
[`setup_smoke.json`](evidence/issue_1111_carla_setup_smoke_2026-05-18/setup_smoke.json).

## Artifact Decision

The raw generated files remain under ignored `output/carla_bridge/` and are reproducible from the
tracked scenario config, commit, and commands above. Compact copies were promoted into
`docs/context/evidence/issue_1111_carla_setup_smoke_2026-05-18/` because they are small,
reviewable, and support the issue handoff. No model checkpoint, benchmark bundle, video, or CARLA
runtime artifact was generated.
