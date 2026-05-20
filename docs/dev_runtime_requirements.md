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

## GitHub And CI-Parity Tools

These are not Python dependencies, but they make local validation and PR work match the repository
workflow:

* `gh` - GitHub issue, PR, checks, and Actions log workflows.
* `jq` - JSON validation in CI smoke paths and local diagnostics.
* `ffmpeg` - video/rendering paths and GitHub CI parity.

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
* SLURM/Auxme access via `sbatch`.
* Whether long training jobs are allowed locally or must run in tmux/SLURM.
* Local artifact/cache locations and durable upload expectations.

Use `docs/templates/local.machine.example.md` as the template. Do not commit secrets or
machine-specific overrides.

## Optional Python Extras And External Artifacts

Some workflows require Python extras or external assets in addition to host tools:

* ORCA/RVO2 planner paths may require `uv sync --extra orca`.
* Imitation workflows use `uv sync --group imitation` and `uv run --group imitation ...`.
* RLlib/DreamerV3 paths use `uv sync --extra rllib` and `uv run --extra rllib ...`.
* Social-Navigation-PyEnvs adapter experiments require the external checkout or a documented
  fail-closed `not_available`/`failed` status.
* CARLA live replay requires a compatible CARLA installation or Docker image and must report
  boundary limitations explicitly.

Do not silently rely on local `output/` contents for durable dependencies. Promote required
artifacts to a durable source and track only small manifests or pointers in git.
