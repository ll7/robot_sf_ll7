# robot-sf

> 2024-03-28 Under Development. See <https://github.com/ll7/robot_sf_ll7/issues>.

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

#### Complete Test Suite

Run all tests for both robot-sf and fast-pysf:

```sh
# Run comprehensive test suite (both main and fast-pysf tests)
./scripts/run_all_tests.sh
```

#### Main Robot-SF Tests

```sh
# Run main robot-sf tests (170 tests)
uv run pytest tests/ -v

# Run with linting
uv run pytest tests/
uv run pylint robot_sf
```

#### PySocialForce Tests

The PySocialForce tests are located in the `fast-pysf/tests/` directory and include comprehensive coverage (121 tests):

```sh
# Run fast-pysf tests
cd fast-pysf
uv run python -m pytest tests/ -v
```

Or with dev dependencies explicitly:

```sh
cd fast-pysf  
uv run --extra dev python -m pytest tests/ -v
```

The fast-pysf test suite includes:
- **Configuration tests**: All force and simulation config classes
- **Force calculation tests**: Desired, social, and group forces
- **Navigation tests**: Route navigation and waypoint management
- **Pedestrian grouping tests**: State management and group dynamics
- **Scene tests**: Pedestrian states and environment obstacles
- **Map loading tests**: Various map formats and edge cases
- **Simulator functionality tests**: Core simulation logic

#### Code Quality

Both projects now include comprehensive linting with Ruff:

```sh
# Check code quality for main project
uv run ruff check .
uv run ruff format .

# Check code quality for fast-pysf
cd fast-pysf
uv run ruff check .
uv run ruff format .
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
