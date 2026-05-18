# Issue #1179 CARLA Docker Runtime

Issue: [#1179](https://github.com/ll7/robot_sf_ll7/issues/1179)
Parent epic: [#872](https://github.com/ll7/robot_sf_ll7/issues/872)
Downstream live replay: [#1169](https://github.com/ll7/robot_sf_ll7/issues/1169)
Runner prerequisite: [#1119](https://github.com/ll7/robot_sf_ll7/issues/1119)

## Decision

Add an opt-in CARLA Docker runtime interface for the pinned image
`carlasim/carla:0.9.16`. The interface keeps normal Robot-SF and CARLA T0/T1 import paths
CARLA-free while giving maintainers a deterministic preflight and lifecycle command for a Linux
x86_64 NVIDIA Docker host.

The runtime command checks prerequisites before starting CARLA and reports `not-available` instead
of treating missing Docker, GPU, NVIDIA Container Toolkit, ports, image storage, or Python API
support as successful simulator evidence.

## Implemented Path

* Runtime helper: `robot_sf_carla_bridge/docker_runtime.py`
* CLI: `robot-sf-carla-docker-runtime`
* Package script: `pyproject.toml`
* Contract tests: `tests/carla_bridge/test_docker_runtime.py`

Recommended command sequence on a CARLA-capable host:

```bash
uv run robot-sf-carla-docker-runtime preflight --json
uv run robot-sf-carla-docker-runtime smoke --pull --json
```

The `smoke --pull` path reuses `carlasim/carla:0.9.16` when present, otherwise it checks Docker
storage before pulling. It requires at least `50 GiB` free and warns below `80 GiB`. The smoke path
starts the server container with explicit `2000-2002` port mappings, connects the host-side Python
client, records client/server version and map metadata, captures a Docker log tail, and stops the
container in success and failure paths.

## Boundary

This is runtime substrate evidence only. It does not replay Robot-SF scenarios, spawn actors from
T0 payloads, compare metrics, run training, or prove CARLA transfer. True replay semantics remain
with #1169.

The default client placement remains Option A from #1179: server in Docker, Python client from the
Robot-SF repo command path using `carla==0.9.16` or a matching bundled wheel. Option B, a companion
client container, is still only a fallback if host-side Python compatibility fails on the target
runner.

## Local Validation

Passing CARLA-free contract checks:

```bash
rtk uv run pytest tests/carla_bridge/test_runtime.py tests/carla_bridge/test_t1_replay_smoke.py tests/carla_bridge/test_docker_runtime.py -q
```

Result: `14 passed`.
Post-review rerun after Docker runtime robustness fixes: `18 passed`.
Second post-review rerun after image validation, CUDA pull, image metadata, and startup-cleanup
hardening: `22 passed`.

Broader CARLA bridge regression suite:

```bash
rtk uv run pytest tests/carla_bridge -q
```

Result: `71 passed`.
Post-review rerun after Docker runtime robustness fixes: `75 passed`.
Second post-review rerun after image validation, CUDA pull, image metadata, and startup-cleanup
hardening: `79 passed`.

Post-merge PR readiness gate:

```bash
rtk bash -lc 'PYTEST_NUM_WORKERS=8 BASE_REF=origin/main scripts/dev/pr_ready_check.sh'
```

Result: Ruff passed, the full pytest suite reported `3702 passed, 10 skipped`, and the touched
definition TODO ratchet passed.
Post-review rerun after Docker runtime robustness fixes: `3706 passed, 10 skipped`, with the
touched-definition TODO ratchet still passing.
Second post-review rerun after image validation, CUDA pull, image metadata, and startup-cleanup
hardening: `3710 passed, 10 skipped`, with the touched-definition TODO ratchet still passing.

Docs proof consistency:

```bash
rtk bash -lc 'BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh'
```

Result: passed for the changed docs files.

Host-side Python package resolver check:

```bash
rtk uv pip install --dry-run 'carla==0.9.16'
```

Result: uv resolved the package for this Python 3.12.3 environment and reported it would install
`carla==0.9.16`. The package was not installed into the normal repo environment.

Local runtime preflight:

```bash
rtk uv run robot-sf-carla-docker-runtime preflight --skip-api-check --json
```

Observed on `Linux x86_64` with an NVIDIA GeForce RTX 3080: status `not-available`,
`missing_capability: docker-daemon`, because Docker Engine 29.4.2 client reports `Server: null` and
fails with permission denied while connecting to `unix:///var/run/docker.sock`.

This local result is a precise runtime blocker, not live CARLA proof. A Docker/NVIDIA-capable host
still needs to run `robot-sf-carla-docker-runtime smoke --pull --json` to record image digest,
client/server versions, active map, Docker command, status, and log tail.

Generated `output/coverage/` files from validation remain ignored and disposable. No generated
CARLA Docker image, benchmark bundle, model checkpoint, or durable runtime artifact was produced on
this host.
