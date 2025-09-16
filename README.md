# robot-sf

> 2025-09-16 Under Development. See <https://github.com/ll7/robot_sf_ll7/issues>.

## About

This project provides a training environment for the simulation of a robot moving
in a pedestrian-filled space.

The project interfaces with Faram Foundations "Gymnasium" (former OpenAI Gym)
to facilitate trainings with various
SOTA reinforcement learning algorithms like e.g. StableBaselines3.
For simulating the pedestrians, the SocialForce model is used via a dependency
on a fork of PySocialForce.

Following video outlines some training results where a robot with e-scooter
kinematics is driving at the campus of University of Augsburg using real
map data from OpenStreetMap.

![](./docs/video/demo_01.gif)

## Development

Refer to the [development guide](./docs/dev_guide.md) for contribution guidelines, code standards, and templates.

## Installation

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

### Development Setup

For development work with additional tools:

```sh
# Install with development dependencies
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest tests

# Run linting and formatting
uv run ruff check .
uv run ruff format .
```

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

#### Docker Installation (Advanced)

For containerized environments:

```sh
docker compose build && docker compose run \
    robotsf-cuda python ./scripts/training_ppo.py
```

*Note: See [GPU setup documentation](./docs/GPU_SETUP.md) for Docker with GPU support.*

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

#### PySocialForce Tests

The PySocialForce tests are located in the `fast-pysf/tests/` directory and can be run with:

```sh
cd fast-pysf
uv run python -m pytest tests/ -v
```

Or with dev dependencies explicitly:

```sh
cd fast-pysf  
uv run --extra dev python -m pytest tests/ -v
```

All tests should pass successfully. The test suite includes:
- Force calculation tests (desired, social, group repulsion forces)
- Map loading tests 
- Simulator functionality tests

#### Run Linter / Tests

```sh
pytest tests
pylint robot_sf
```

#### GUI Tests

```sh
pytest test_pygame
```

### 5. Run Visual Debugging of Pre-Trained Demo Models

```sh
python3 examples/demo_offensive.py
python3 examples/demo_defensive.py
```

[Visualization](./docs/SIM_VIEW.md)

### 6. Run StableBaselines Training (Docker)

```sh
docker compose build && docker compose run \
    robotsf-cuda python ./scripts/training_ppo.py
```

*Note: See [this setup](./docs/GPU_SETUP.md) to install Docker with GPU support.*

> Older versions use `docker-compose` instead of `docker compose`.

### 7. Edit Maps

The preferred way to create maps: [SVG Editor](./docs/SVG_MAP_EDITOR.md)

### 8. Optimize Training Hyperparams (Docker)

```sh
docker-compose build && docker-compose run \
    robotsf-cuda python ./scripts/hparam_opt.py
```

### 9. Extension: Pedestrian as Adversarial-Agent

The pedestrian is an adversarial agent who tries to find weak points in the vehicle's policy.

The Environment is built according to gymnasium rules, so that multiple RL algorithms can be used to train the pedestrian.

It is important to know that the pedestrian always spawns near the robot.

![demo_ped](./docs/video/demo_ped.gif)

```sh
python3 examples/demo_pedestrian.py
```

[Visualization](./docs/SIM_VIEW.md)

## 📚 Documentation

### Core Documentation
- [Environment Refactoring](./docs/refactoring/) - **NEW**: Comprehensive guide to the refactored environment architecture
- [Data Analysis](./docs/DATA_ANALYSIS.md) - Analysis tools and utilities
- [GPU Setup](./docs/GPU_SETUP.md) - GPU configuration for training
- [Map Editor Usage](./docs/MAP_EDITOR_USAGE.md) - Creating and editing simulation maps
- [SVG Map Editor](./docs/SVG_MAP_EDITOR.md) - SVG-based map creation
- [Simulation View](./docs/SIM_VIEW.md) - Visualization and rendering
- [UV Migration](./docs/UV_MIGRATION.md) - Migration to UV package manager

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

Core capabilities:

```text
recompute_snqi_weights.py      # Strategy comparison (default, safety, efficiency, pareto) + normalization checks
snqi_weight_optimization.py    # Grid + differential evolution + optional sensitivity
snqi_sensitivity_analysis.py   # Deeper robustness & interaction analysis
```

Pass `--seed` for deterministic optimization/sampling. All outputs embed a `_metadata` block with schema version, git commit, seed, and provenance for reproducibility. A unified CLI subcommand (`robot_sf_bench snqi ...`) is planned.
Pass `--seed` for deterministic optimization/sampling. All outputs embed a `_metadata` block with schema version, git commit, seed, and provenance for reproducibility. A unified CLI subcommand (`robot_sf_bench snqi ...`) is available.

Inline SNQI during episode generation (compute `metrics.snqi` on the fly):

```sh
uv run robot_sf_bench run \
    --matrix configs/baselines/example_matrix.yaml \
    --out results/episodes.jsonl \
    --schema docs/dev/issues/social-navigation-benchmark/episode_schema.json \
    --snqi-weights model/snqi_canonical_weights_v1.json \
    --snqi-baseline results/baseline_stats.json
```

