# Robot SF Runtime Requirements

[← Back to Documentation Index](./README.md)

This page tracks runtime prerequisites that are not fully managed by `uv sync --all-extras`.
Python packages belong in `pyproject.toml`; machine tools, daemons, host limits, and optional
external services belong here or in the local-only `local.machine.md`.

Check the current machine with:

```bash
scripts/dev/check_runtime_requirements.sh
```

Use strict mode when you want CI-parity recommendations to fail fast:

```bash
scripts/dev/check_runtime_requirements.sh --strict
```

For pasteable setup diagnostics in issue reports, run:

```bash
uv run robot_sf_bench doctor
```

The doctor command reports Python/package source details, required and optional host tools,
headless rendering environment variables, artifact-root writability, and a minimal reset/step
environment smoke. It is read-only apart from temporary probe files and reports missing optional
tools as warnings instead of attempting automatic repair.

## Core Tools

These are expected for normal development:

* `git` - checkout, branch, and worktree management.
* `uv` - dependency sync and Python command runner.
* Python 3.12 - the supported interpreter version in GitHub Actions. Run project commands through
  `uv run ...` rather than a globally activated interpreter.

After cloning or creating a fresh linked worktree:

```bash
uv sync --all-extras
source .venv/bin/activate
```

CARLA is deliberately excluded from `--all-extras`. Use `uv sync --all-extras --group carla` only
on machines or worktrees that need the host-side CARLA Python client.

## GitHub And CI-Parity Tools

These are not Python dependencies, but they make local validation and PR work match the repository
workflow:

* `gh` - GitHub issue, PR, checks, and Actions log workflows.
* `jq` - JSON validation in CI smoke paths and local diagnostics.
* `ffmpeg` - video/rendering paths and GitHub CI parity.
* `gh-signoff` - optional advisory local-CI statuses. `scripts/dev/local_signoff.sh`
  auto-installs `basecamp/gh-signoff` when needed and never changes branch protection.

GitHub Actions currently installs these Ubuntu packages for headless jobs:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 fonts-dejavu-core jq
```

The promoted-planner and nightly performance workflows use the same headless stack, without `jq`
where it is not needed.

## Headless Rendering

Use these environment values for local GUI/rendering tests on headless machines:

```bash
DISPLAY=
MPLBACKEND=Agg
SDL_VIDEODRIVER=dummy
```

The CI workflows also set `PYTHONUNBUFFERED=1` and `PYGAME_HIDE_SUPPORT_PROMPT=1`.

## Docker And gh-act

Docker is optional for ordinary Python validation, but required for:

* `gh act ...` local GitHub Actions experiments,
* benchmark Docker reproduction,
* CARLA Docker runtime paths,
* GPU Docker repro checks when `--gpus all` is part of the proof.

`gh-act` is installed as a GitHub CLI extension:

```bash
gh extension install https://github.com/nektos/gh-act
```

Current status from issue #1342:

* `gh-act` installed successfully as `gh act` v0.2.88 on the maintainer machine.
* `gh act -l pull_request` listed the pull-request jobs.
* `gh act pull_request -j ci --dryrun -P ubuntu-latest=catthehacker/ubuntu:act-latest` passed
  graph/step validation.
* `gh act pull_request -j promoted-planner-smoke --dryrun -P ubuntu-latest=catthehacker/ubuntu:act-latest`
  passed graph/step validation.
* Running without `-P ...` prompts for a default image and fails in non-interactive automation.

Treat `gh act` as experimental until a real narrow job, not only `--dryrun`, is proven and recorded.
It does not replace GitHub CI or:

```bash
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

## Optional Machine Capabilities

Record machine-specific availability and limits in local-only `local.machine.md`:

* CPU worker caps and memory limits.
* GPU availability, `nvidia-smi`, and NVIDIA Docker support.
* CARLA Docker availability, including Docker daemon access, NVIDIA Container Toolkit status, the
  pinned CARLA image, host ports, and the command that last proved connectivity.
* SLURM/Auxme access via `sbatch`.
* Whether long training jobs are allowed locally or must run in tmux/SLURM.
* Local artifact/cache locations and durable upload expectations.

Use `docs/templates/local.machine.example.md` as the template. Do not commit secrets or
machine-specific overrides. In linked worktrees, prefer symlinking `local.machine.md` from the main
checkout so Docker/GPU/CARLA status stays current across worktrees; copy it only when the worktree
truly needs different machine-specific behavior.

## Optional Python Extras And External Artifacts

Some workflows require Python extras or external assets in addition to host tools:

* ORCA/RVO2 planner paths may require `uv sync --extra orca`.
* Training and optimization workflows use `uv sync --extra training` and
  `uv run --extra training ...`.
* GPU training workflows should combine training and GPU extras, for example
  `uv sync --extra training --extra gpu`.
* Imitation workflows use `uv sync --group imitation` and `uv run --group imitation ...`.
* RLlib/DreamerV3 paths use `uv sync --extra rllib --extra training` and
  `uv run --extra rllib --extra training ...`.
* Social-Navigation-PyEnvs adapter experiments require the external checkout or a documented
  fail-closed `not_available`/`failed` status.
* CARLA live replay requires both a compatible CARLA server path and the matching host-side Python
  client. The host-side client is pinned in the `carla` dependency group rather than an optional
  extra, so `uv sync --all-extras` stays CARLA-free:

  ```bash
  uv sync --all-extras --group carla
  scripts/dev/check_carla_runtime.sh
  scripts/dev/check_carla_runtime.sh --smoke
  ```

  Use `scripts/dev/check_carla_runtime.sh --pull` or
  `scripts/dev/check_carla_runtime.sh --smoke --pull` only when the host may download the pinned
  `carlasim/carla:0.9.16` image. For a minimal CARLA-only environment, `uv sync --group carla` is
  also valid, but it does not preserve optional extras from a prior all-extras sync. For one-off
  checks without syncing the group first, `uv run --with carla==0.9.16 ...` remains acceptable, but
  the group-backed script is the reproducible repository path. A Docker image or passing preflight
  is setup evidence only. Treat live replay or parity claims as complete only when the exact replay
  command and boundary limitations are recorded.

Do not silently rely on local `output/` contents for durable dependencies. Promote required
artifacts to a durable source and track only small manifests or pointers in git.
