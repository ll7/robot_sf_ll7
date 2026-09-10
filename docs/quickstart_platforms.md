# Platform Setup Profiles

Short platform paths for the setups the repository already exercises: Linux, macOS, and
headless continuous integration. Windows is **unverified**: nothing below presents it as
supported.

Core setup (all profiles): Python `>=3.11` (repository CI uses 3.11–3.13; see
`pyproject.toml` `requires-python`), the `uv` package manager, and the `robot-sf`
console entry point (`pyproject.toml` `[project.scripts]`).

## Linux

Prerequisites: Python 3.11+, `uv`, a working C compiler toolchain (third-party extensions
such as `third_party/python-rvo2` build from source during install).

```bash
git clone https://github.com/ll7/robot_sf_ll7
cd robot_sf_ll7
scripts/dev/check_runtime_requirements.sh
uv sync --all-extras
```

Verify:

```bash
uv run robot-sf doctor --skip-env-smoke --skip-quickstart-smoke
```

Smoke:

```bash
uv run robot-sf examples run quickstart/01_basic_robot --fast
```

The manifest-backed `quickstart/01_basic_robot` example resolves to
`examples/quickstart/01_basic_robot.py` and runs a short CPU episode.

## macOS

Prerequisites: same as Linux. The repository CI exercises `macos-latest` (Python 3.11/3.13
compatibility matrix), so the standard path is verified there. If a source build fails on
missing compiler headers, install the Xcode Command Line Tools first.

```bash
git clone https://github.com/ll7/robot_sf_ll7
cd robot_sf_ll7
scripts/dev/check_runtime_requirements.sh
uv sync --all-extras
```

Verify:

```bash
uv run robot-sf doctor --skip-env-smoke --skip-quickstart-smoke
```

Smoke:

```bash
uv run robot-sf examples run quickstart/01_basic_robot --fast
```

The macOS path is verified for the compatibility setup; interactive windowing remains outside
the CI coverage described below.

Core setup only: visualization extras install the same way as Linux, but windowing behavior
on macOS is not covered by repository CI beyond the compatibility test subset. If a visualization
workflow needs a window and the host cannot open one, use the headless profile instead.

## Headless (servers and CI)

Prerequisites: same as Linux, plus these environment values so nothing requires an
interactive display (same stack as repository CI; see `docs/dev_runtime_requirements.md`):

```bash
export DISPLAY=
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export PYTHONUNBUFFERED=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
```

Setup:

```bash
git clone https://github.com/ll7/robot_sf_ll7
cd robot_sf_ll7
scripts/dev/check_runtime_requirements.sh
uv sync --all-extras
```

Verify:

```bash
uv run robot-sf doctor --skip-env-smoke --skip-quickstart-smoke
```

Smoke:

```bash
uv run robot-sf examples run quickstart/01_basic_robot --fast
```

The manifest-backed example is the headless first-episode smoke; the exported variables above
keep its rendering dependencies off-screen. GPU, Docker, CARLA, SLURM, model downloads, and
external data are explicitly out of scope here; see `docs/dev_runtime_requirements.md` for
those lanes.

## What is not covered

- **Windows**: unverified — not presented as supported anywhere in this page.
- Optional stacks (visualization extras beyond the headless subset, training, CARLA,
  SLURM, external datasets, model retrieval): see `docs/dev_runtime_requirements.md`.
