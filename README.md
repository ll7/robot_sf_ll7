# robot-sf

> 2025-09-16 Under Development. See <https://github.com/ll7/robot_sf_ll7/issues>.

<!-- This document is the first touchpoint for users and developers of the robot-sf project. Keep it very concise and focused on essential information. -->

## About

This project provides a training environment for the simulation of a robot moving
in a pedestrian-filled space.

The project interfaces with Faram Foundations "Gymnasium" (successor to OpenAI Gym)
to facilitate trainings with various
SOTA reinforcement learning algorithms like e.g. StableBaselines3.
For simulating the pedestrians, the SocialForce model is used via a dependency
on a fork of PySocialForce.

Following video outlines some training results where a robot with e-scooter
kinematics is driving at the campus of University of Augsburg using real
map data from OpenStreetMap.

![](./docs/video/demo_01.gif)

- [About](#about)
- [Attribution and Upstream Work](#attribution-and-upstream-work)
- [Development and Intallation](#development-and-intallation)
  - [Prerequisites](#prerequisites)
  - [Quick Start](#quick-start)
  - [Examples Catalog](#examples-catalog)
  - [Development Setup](#development-setup)
  - [Artifact Outputs](#artifact-outputs)
  - [Alternative Installation Methods](#alternative-installation-methods)
    - [Manual dependency installation](#manual-dependency-installation)
  - [System Dependencies](#system-dependencies)
  - [Tests](#tests)
    - [Unified Test Suite](#unified-test-suite)
    - [Run Linter / Tests](#run-linter--tests)
    - [GUI Tests](#gui-tests)
  - [5. Run Visual Debugging of Pre-Trained Demo Models](#5-run-visual-debugging-of-pre-trained-demo-models)
  - [6. Edit Maps](#6-edit-maps)
  - [7. Extension: Pedestrian as Adversarial-Agent](#7-extension-pedestrian-as-adversarial-agent)
- [📚 Documentation](#-documentation)
  - [Core Documentation](#core-documentation)
  - [Environment Architecture (New!)](#environment-architecture-new)
  - [SNQI Weight Tooling (Benchmark Metrics)](#snqi-weight-tooling-benchmark-metrics)

## Attribution and Upstream Work

This project builds on prior work in social robot navigation, especially:

Caruso, Matteo, Enrico Regolin, Federico Julian Camerota Verdu, Stefano Alberto Russo, Luca
Bortolussi, and Stefano Seriani. "Robot Navigation in Crowded Environments: A Reinforcement
Learning Approach." *Machines* 11, no. 2 (2023): 268.
https://doi.org/10.3390/machines11020268

As stated in the paper's Data Availability Statement, data/videos/samples/code were made
publicly available in:

- https://github.com/EnricoReg/robot-sf
- https://github.com/EnricoReg/asynch-rl
- https://github.com/matteocaruso1993/crowd_nav_experimental

Additional related upstream repositories referenced in this project context:

- https://github.com/Bonifatius94/robot-sf
- https://github.com/yuxiang-gao/PySocialForce
- https://github.com/Bonifatius94/PySocialForce

For fast-pysf (PySocialForce), this project also acknowledges the upstream model and implementation
lineage:

- This project is based on Sven Kreiss's implementation of the vanilla social force model:
  https://github.com/svenkreiss/socialforce
- The implementation of forces drew inspiration from pedsim_ros:
  https://github.com/srl-freiburg/pedsim_ros

Main references for fast-pysf:

- Helbing, D., and P. Molnar. "Social force model for pedestrian dynamics." *Physical Review E*
  51, no. 5 (1995): 4282-4286. https://doi.org/10.1103/PhysRevE.51.4282
- Moussaid, M., N. Perozo, S. Garnier, D. Helbing, and G. Theraulaz. "The walking behaviour of
  pedestrian social groups and its impact on crowd dynamics." *PLoS ONE* 5, no. 4 (2010): 1-7.
  https://doi.org/10.1371/journal.pone.0010047

## Development and Intallation

Refer to the [development guide](./docs/dev_guide.md) for contribution guidelines, code standards, and templates.

This project now uses `uv` for modern Python dependency management and virtual environment handling.

### Prerequisites

Install Python 3.10+ (Python **3.12** is recommended) and `uv`:

```sh
# Install uv (the modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

### Quick Start

```sh
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/ll7/robot_sf_ll7
cd robot_sf_ll7

# Install all dependencies and create virtual environment automatically
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Install system dependencies (Linux/Ubuntu)
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### Examples Catalog

Consult the curated [`examples/README.md`](./examples/README.md) for quickstart,
advanced, benchmark, and plotting workflows. Each entry lists prerequisites,
expected outputs, and CI status.

### Development Setup

For development work with additional tools:

```sh
# Install with development dependencies
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests (unified suite: robot_sf + fast-pysf)
uv run pytest  # → 893 tests total

# Run only robot_sf tests
uv run pytest tests  # → 881 tests

# Run only fast-pysf tests
uv run pytest fast-pysf/tests  # → 12 tests

# Run linting and formatting
uv run ruff check .
uv run ruff format .
```

### Artifact Outputs

- Generated artifacts are routed into the canonical `output/` tree (`output/coverage/`, `output/benchmarks/`, `output/recordings/`, `output/wandb/`, `output/tmp/`).
- After pulling new changes, run `uv run python scripts/tools/migrate_artifacts.py` (or `uv run robot-sf-migrate-artifacts`) to relocate any legacy `results/`, `recordings/`, or `htmlcov/` directories.
- Use `uv run python scripts/tools/check_artifact_root.py` to verify the repository root stays clean; CI runs the same guard.
- Set `ROBOT_SF_ARTIFACT_ROOT=/path/to/custom/output` before invoking scripts if you need to direct artifacts elsewhere—the helpers and guard respect the override.
- Coverage HTML is available via `uv run python scripts/coverage/open_coverage_report.py`, which opens `output/coverage/htmlcov/index.html` cross-platform.

### Alternative Installation Methods

#### Manual dependency installation

If you prefer more control over the installation:

```sh
# Create virtual environment with specific Python version
uv venv --python 3.12

# Activate environment
source .venv/bin/activate

# Install project in editable mode
uv sync

# Install development tools (optional)
uv sync --group=dev
```

### System Dependencies

**FFMPEG** (required for video recording):

```sh
# Ubuntu/Debian
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Tests

#### Unified Test Suite

The project uses a unified test suite that runs both robot_sf and fast-pysf tests via a single command:

```sh
# Run all tests (recommended)
uv run pytest  # → 893 tests (881 robot_sf + 12 fast-pysf)

# Run only robot_sf tests
uv run pytest tests  # → 881 tests

# Run only fast-pysf tests  
uv run pytest fast-pysf/tests  # → 12 tests

# Run with parallel execution (faster)
uv run pytest -n auto
```

All tests should pass successfully. The test suite includes:
- **robot_sf tests** (881): Unit, integration, baselines, benchmarks
- **fast-pysf tests** (12): Force calculations, map loading, simulator functionality

**Note**: The fast-pysf tests are now integrated into the main pytest configuration and no longer require running from the `fast-pysf/` directory or using `python -m pytest`.

#### Run Linter / Tests

```sh
# Lint and format
uv run ruff check --fix . && uv run ruff format .

# Run all tests (unified suite)
uv run pytest

# Legacy linter (for comparison)
pylint robot_sf
```

#### GUI Tests

```sh
pytest test_pygame
```

### 5. Run Visual Debugging of Pre-Trained Demo Models

```sh
uv run python examples/advanced/10_offensive_policy.py
uv run python examples/advanced/09_defensive_policy.py
# Classic interactions deterministic PPO visualization (Feature 128)
uv run python examples/_archived/classic_interactions_pygame.py
```

[Visualization](./docs/SIM_VIEW.md)

### 6. Edit Maps

The preferred way to create maps: [SVG Editor](./docs/SVG_MAP_EDITOR.md)

### 7. Extension: Pedestrian as Adversarial-Agent

The pedestrian is an adversarial agent who tries to find weak points in the vehicle's policy.

The Environment is built according to gymnasium rules, so that multiple RL algorithms can be used to train the pedestrian.

It is important to know that the pedestrian always spawns near the robot.

![demo_ped](./docs/video/demo_ped.gif)

```sh
uv run python examples/advanced/06_pedestrian_env_factory.py
```

[Visualization](./docs/SIM_VIEW.md)

## 📚 Documentation

**Comprehensive documentation is available in the [`docs/`](./docs/) directory.**

For detailed guides on setup, development, benchmarking, and architecture, visit the **[Documentation Index](./docs/README.md)**.

### Core Documentation
- [Environment Refactoring](./docs/refactoring/) - **NEW**: Comprehensive guide to the refactored environment architecture
- [Data Analysis](./docs/DATA_ANALYSIS.md) - Analysis tools and utilities
- [Map Editor Usage](./docs/MAP_EDITOR_USAGE.md) - Creating and editing simulation maps
- [SVG Map Editor](./docs/SVG_MAP_EDITOR.md) - SVG-based map creation
- [Simulation View](./docs/SIM_VIEW.md) - Visualization and rendering
- [UV Migration](./docs/UV_MIGRATION.md) - Migration to UV package manager
- [Artifact Policy Quickstart](./specs/243-clean-output-dirs/quickstart.md) - Migration workflow, guard usage, and override instructions for the canonical `output/` tree
 - [Contributing Agents Guide](./AGENTS.md) – Repository structure, coding style, test workflow, and contributor conventions (start here if new!)

### Environment Architecture (New!)
The project has been refactored to provide a **consistent, extensible environment system**:

```python
# New factory pattern for environment creation
from robot_sf.gym_env.environment_factory import (
    make_robot_env,
    make_image_robot_env, 
    make_pedestrian_env
)

# Clean, consistent interface
robot_env = make_robot_env(debug=True)
image_env = make_image_robot_env(debug=True)
ped_env = make_pedestrian_env(robot_model=model, debug=True)
```

**Key Benefits:**
- ✅ **50% reduction** in code duplication
- ✅ **Consistent interface** across all environment types
- ✅ **Easy extensibility** for new environment types
- ✅ **Backward compatibility** maintained

📖 **[Read the full refactoring documentation →](./docs/refactoring/)**

### SNQI Weight Tooling (Benchmark Metrics)

Tools for recomputing, optimizing, and analyzing Social Navigation Quality Index (SNQI) weights are now available:

- User Guide: [`docs/snqi-weight-tools/README.md`](./docs/snqi-weight-tools/README.md)
- Design & architecture: [`docs/dev/issues/snqi-recomputation/DESIGN.md`](./docs/dev/issues/snqi-recomputation/DESIGN.md)
- Headless usage (minimal deps): see [Headless mode](./docs/snqi-weight-tools/README.md#headless-mode-minimal-deps)
