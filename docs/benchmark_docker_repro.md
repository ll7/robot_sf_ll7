# Benchmark Docker Reproduction Path

Issue: [#1086](https://github.com/ll7/robot_sf_ll7/issues/1086)

This page documents the pinned Docker path for a small canonical benchmark verification slice. It
is intended for reviewers who want to verify that the benchmark CLI, scenario manifest, episode
writer, and aggregate writer execute in a fresh headless container without bespoke local setup.

## One-Command Smoke

From the repository root:

```bash
scripts/repro/run_benchmark_docker_smoke.sh
```

The wrapper builds `docker/benchmark-repro.Dockerfile` and runs
`scripts/repro/benchmark_bundle_smoke.sh` inside the image.

Default image tag:

```text
robot-sf-benchmark-repro:py312.3-uv0.11.9
```

## GitHub Actions Runner Smoke

The path-limited workflow `.github/workflows/benchmark-docker-repro-smoke.yml` runs the same
wrapper when Docker reproduction files change, and it can also be triggered manually with
`workflow_dispatch`.

The workflow records runner qualification evidence before building the image:

* runner OS and architecture
* Docker daemon version and `docker info`
* `nvidia-smi` availability
* `docker run --gpus all ... nvidia-smi` availability

GitHub-hosted `ubuntu-latest` can provide Docker daemon proof for the CPU/headless smoke. It is not
expected to provide NVIDIA GPU evidence; the workflow records that as an explicit not-available
condition rather than treating it as benchmark success.

## What Runs

The smoke uses:

* matrix: `configs/scenarios/planner_sanity_matrix_v1.yaml`
* algorithm: `goal`
* repeats: `1`
* horizon: `300`
* workers: `1`
* video: disabled

The script executes these benchmark surfaces:

```bash
uv run robot_sf_bench validate-config --matrix configs/scenarios/planner_sanity_matrix_v1.yaml
uv run robot_sf_bench preview-scenarios --matrix configs/scenarios/planner_sanity_matrix_v1.yaml
uv run robot_sf_bench run --matrix configs/scenarios/planner_sanity_matrix_v1.yaml --algo goal
uv run robot_sf_bench aggregate --in output/docker_repro/benchmark_bundle_smoke/episodes.jsonl
```

The exact command flags are versioned in `scripts/repro/benchmark_bundle_smoke.sh`.

## Output Artifacts

The default output directory is:

```text
output/docker_repro/benchmark_bundle_smoke/
```

Expected files:

* `validate_config.json` and `validate_config.log`
* `preview_scenarios.json` and `preview_scenarios.log`
* `run_summary.json` and `run.log`
* `episodes.jsonl`
* `summary.json`
* `aggregate.log`
* `manifest.json`

`manifest.json` records the repository commit, matrix, algorithm, smoke parameters, episode count,
artifact sizes, and SHA-256 checksums. `output/` remains local and ignored; rerun the smoke to
regenerate these artifacts from the tracked Dockerfile, scripts, configs, and source commit.

## Version Pinning

The container recipe pins the main reproduction surfaces:

* base image tag: `python:3.12.3-slim-bookworm`
* uv package: `0.11.9`
* Python dependencies: `uv.lock` with `uv sync --all-extras --frozen`
* benchmark source: the checked-out repository commit copied into the image
* benchmark matrix: `configs/scenarios/planner_sanity_matrix_v1.yaml`

For release-grade archival, record the built image digest alongside the repository commit and
`manifest.json` checksums. The tracked Dockerfile is the source build recipe; the repository does
not currently publish a prebuilt image.

## Limits

This path is a smoke verification path, not a full campaign runner.

* It does not claim GPU determinism.
* It does not run the full paper or camera-ready benchmark campaign.
* It runs headless CPU execution only.
* It is expected to verify that the canonical small slice runs and writes inspectable artifacts,
  not that every floating-point metric is bit-identical across host CPUs.
