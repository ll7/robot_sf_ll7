# Issue #1086 Docker Reproduction Path

Issue: [#1086](https://github.com/ll7/robot_sf_ll7/issues/1086)
Follow-up: [#1119](https://github.com/ll7/robot_sf_ll7/issues/1119)

## Decision

Add a tracked Docker build recipe plus a one-command wrapper rather than publishing a prebuilt image
in this first pass. The build recipe is reviewable in git, ties dependency resolution to `uv.lock`,
and avoids creating a new external image publication process before the benchmark release workflow
has an image registry decision.

## Implemented Path

* Dockerfile: `docker/benchmark-repro.Dockerfile`
* Host wrapper: `scripts/repro/run_benchmark_docker_smoke.sh`
* In-container smoke: `scripts/repro/benchmark_bundle_smoke.sh`
* User guide: `docs/benchmark_docker_repro.md`

The smoke targets `configs/scenarios/planner_sanity_matrix_v1.yaml` with the `goal` planner,
`repeats=1`, `horizon=300`, `workers=1`, and video disabled. This keeps runtime small while proving
the scenario validation, preview, episode JSONL generation, and aggregation surfaces.

## Artifact Policy

The generated path is `output/docker_repro/benchmark_bundle_smoke/`. These files are reproducible
from tracked sources and remain ignored/local:

* `validate_config.json`
* `preview_scenarios.json`
* `run_summary.json`
* `episodes.jsonl`
* `summary.json`
* `manifest.json`
* companion logs

No durable artifact upload is required for the smoke itself. If the path is later promoted to a
paper-facing release artifact, the built image digest and generated manifest checksums should be
published with the release bundle.

## Determinism Boundary

The path pins Python, uv, locked Python dependencies, source commit, scenario matrix, planner,
horizon, repeats, and worker count. It intentionally does not claim GPU determinism, full campaign
coverage, or bit-identical floating-point behavior across CPU architectures.

## Validation Plan

Required checks for this branch:

* `bash -n scripts/repro/benchmark_bundle_smoke.sh scripts/repro/run_benchmark_docker_smoke.sh`
* targeted contract tests for the Dockerfile, scripts, docs, and context note
* direct smoke script execution on the host
* Docker wrapper execution when Docker is available
* full PR readiness after syncing with `origin/main`

## 2026-05-09 Local Validation

Observed locally:

* `bash -n scripts/repro/benchmark_bundle_smoke.sh scripts/repro/run_benchmark_docker_smoke.sh`
  passed.
* `rtk uv run pytest tests/repro/test_benchmark_docker_repro_contract.py -q` passed.
* `ROBOT_SF_REPRO_OUTPUT_ROOT=output/docker_repro/host_smoke_success
  scripts/repro/benchmark_bundle_smoke.sh` passed and wrote three episode records plus
  `summary.json` and `manifest.json`; the aggregate reported `success.mean=1.0` and
  `collisions.mean=0.0` for the `goal` planner.

Docker daemon validation was blocked on this host:

```text
ERROR: permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

The current user is not in the `docker` group for `/var/run/docker.sock`. The Docker build/run
command remains the required final environment proof on a Docker-capable host or CI runner; this is
tracked by follow-up issue #1119.
