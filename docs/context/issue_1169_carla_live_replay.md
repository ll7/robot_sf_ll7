# Issue #1169 CARLA Live T1 Oracle Replay

Issue: [#1169](https://github.com/ll7/robot_sf_ll7/issues/1169)
Parent epic: [#872](https://github.com/ll7/robot_sf_ll7/issues/872)
Runtime substrate: [#1179](https://github.com/ll7/robot_sf_ll7/issues/1179)
Setup-only predecessor: [#1111](https://github.com/ll7/robot_sf_ll7/issues/1111)
Static-geometry follow-up: [#1329](https://github.com/ll7/robot_sf_ll7/issues/1329)
Evidence bundle:
[`docs/context/evidence/issue_1169_carla_live_replay_2026-05-18/`](evidence/issue_1169_carla_live_replay_2026-05-18/)

## Outcome

Issue #1169 adds an opt-in Docker-backed `live-replay` path for the CARLA T1 oracle replay bridge.
The command starts the pinned `carlasim/carla:0.9.16` server, waits for the host-side
`carla==0.9.16` Python client to connect, selects one certified T0 export payload, and attempts to
spawn/replay Robot-SF actors by oracle transforms before stopping the container.

The live replay runner is deliberately fail-closed. The original #1169 slice supported one robot
actor plus scripted pedestrian actors moved by oracle transforms and rejected all static geometry.
The #1329 follow-up branch adds the first bounded static-geometry slice: axis-aligned rectangular
polygon obstacles are spawned as CARLA static-prop proxy actors before the dynamic replay begins,
while unsupported obstacle shapes still fail closed with explicit counts.

## Implemented Path

* Live replay helper: `robot_sf_carla_bridge/live_replay.py`
* Docker wrapper: `robot_sf_carla_bridge/docker_runtime.py`
* CLI: `robot-sf-carla-docker-runtime live-replay`
* Contract tests:
  * `tests/carla_bridge/test_t1_live_replay.py`
  * `tests/carla_bridge/test_docker_runtime.py`

Recommended command on a CARLA-capable host:

```bash
uv run --with carla==0.9.16 robot-sf-carla-docker-runtime live-replay \
  --manifest docs/context/evidence/issue_1111_carla_setup_smoke_2026-05-18/manifest.json \
  --max-steps 10 \
  --pull \
  --json
```

## Local Live Evidence

On May 18, 2026, the host had Docker Engine `29.4.2`, NVIDIA Container Toolkit support, and an
NVIDIA GeForce RTX 3080 visible through Docker.

Initial preflight and no-pull live replay attempts failed closed with:

```text
missing_capability: carla-image
reason: carlasim/carla:0.9.16 is not present locally; rerun with --pull to fetch the pinned image
```

After pulling the pinned image, `live-replay` started CARLA and connected the Python client:

```text
status: connected
client_version: 0.9.16
server_version: 0.9.16
map: Carla/Maps/Town10HD_Opt
image_digest: sha256:aaf1df22702780ece072069e23d03c4879b002ae028c79744b09c4c7ddbae953
```

The replay then failed closed before actor spawning because the inherited #1111 payload
`pr_promoted_planner_smoke` contains four static obstacles:

```text
status: failed
mode: failed
reason: T0 payload static obstacle replay is not implemented
unsupported.static_obstacle_count: 4
cleanup.returncode: 0
```

This is live CARLA execution evidence and a concrete semantic boundary, not a Robot-SF/CARLA parity
claim.

## Issue #1329 Static-Geometry Update

The #1329 implementation removes the blanket static-obstacle rejection for the rectangular polygon
walls present in the #1111 `pr_promoted_planner_smoke` payload. The local proof is still CARLA-free:
fake-CARLA tests verify that rectangular static obstacles are spawned before robot/pedestrian
actors, unsupported static obstacle shapes fail closed before replay, and cleanup destroys static
proxies together with dynamic actors.

The required live host rerun has not been executed on this laptop. The next live proof must run the
same Docker-backed command on a CARLA-capable Linux/NVIDIA host and confirm that the previous
failure reason is no longer `T0 payload static obstacle replay is not implemented`.

## Boundary

The live replay path does not yet provide:

* sensor/perception replay,
* metric parity,
* benchmark-strength CARLA transfer,
* long-running campaign evidence.

Static geometry is currently limited to axis-aligned rectangular polygon proxies. Any unsupported
or malformed static obstacle must remain `failed`, not `oracle-replay`, because silently ignoring
obstacles would weaken the certified T0 scenario contract.

## Validation

Test-first proof:

```bash
uv run pytest tests/carla_bridge/test_t1_live_replay.py -q
```

Initial result before implementation: failed with `ModuleNotFoundError` for
`robot_sf_carla_bridge.live_replay`.

Targeted green checks:

```bash
uv run pytest tests/carla_bridge/test_t1_live_replay.py tests/carla_bridge/test_docker_runtime.py -q
```

Result: `20 passed`.

Full CARLA bridge regression suite:

```bash
uv run pytest tests/carla_bridge -q
```

Result: `84 passed`.

Post-fix focused wrapper check:

```bash
uv run pytest tests/carla_bridge/test_docker_runtime.py::test_docker_live_replay_runs_replay_before_cleanup -q
```

Result: `1 passed`.

Lint:

```bash
uv run ruff check robot_sf_carla_bridge tests/carla_bridge/test_t1_live_replay.py tests/carla_bridge/test_docker_runtime.py
```

Result: all checks passed.

Docs proof consistency:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```

Result: passed for 14 changed files.

Post-`origin/main` PR-readiness gate:

```bash
PYTEST_NUM_WORKERS=8 BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

Result: Ruff passed, formatting reported `1253 files left unchanged`, full pytest reported
`3714 passed, 10 skipped`, and the TODO-docstring backlog ratchet passed for 219 files and 1969
occurrences.

## Artifact Decision

Raw generated files remain ignored under `output/carla_bridge/issue1169_live_replay/` and
`output/coverage/`. The compact JSON summaries copied into
`docs/context/evidence/issue_1169_carla_live_replay_2026-05-18/` are small, reviewable, and support
the issue handoff. The pulled Docker images are local machine state, not repository artifacts.
