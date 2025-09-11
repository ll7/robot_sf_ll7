# Robot SF - Robotics Simulation Environment

Robot SF is a Python-based robotics simulation environment for training robots to navigate in pedestrian-filled spaces using reinforcement learning. The project uses gymnasium (OpenAI Gym) interface with StableBaselines3 for RL training and PySocialForce for pedestrian simulation.

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Bootstrap and Build the Repository
```bash
# Clone with submodules (CRITICAL - contains fast-pysf dependency)
git clone --recurse-submodules https://github.com/ll7/robot_sf_ll7
cd robot_sf_ll7

# Initialize submodules if not already done
git submodule update --init --recursive

# Install uv (modern Python package manager)
pip install uv
# OR: curl -LsSf https://astral.sh/uv/install.sh | sh

# Install system dependencies (required for video recording)
sudo apt-get update && sudo apt-get install -y ffmpeg

# Install all dependencies and create virtual environment
uv sync
# NEVER CANCEL: Takes 2-3 minutes to complete. Set timeout to 5+ minutes.

# Activate virtual environment
source .venv/bin/activate
```

### Development Setup
```bash
# Install with development dependencies
uv sync --extra dev

# Install pre-commit hooks (recommended)
uv run pre-commit install

# Verify installation works
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; print('Import successful')"
```

### Testing
```bash
# Run main test suite
uv run pytest tests
# NEVER CANCEL: Takes 2-3 minutes (170 tests). Set timeout to 5+ minutes.

# Run GUI tests (requires display)
uv run pytest test_pygame
# NEVER CANCEL: Takes 1-2 minutes (3 tests). Set timeout to 3+ minutes.

# Run fast-pysf submodule tests
uv run python -m pytest fast-pysf/tests/ -v
# NOTE: 2 tests may fail due to missing test map files - this is expected.

# Test without GUI (headless mode)
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python examples/demo_defensive.py
```

### Code Quality
```bash
# Run linting (fast)
uv run ruff check .

# Run formatting check (fast)
uv run ruff format --check .

# Run comprehensive linting
uv run pylint robot_sf --exit-zero
# Takes ~16 seconds. Code quality typically scores 9.5+/10.

# Fix formatting automatically
uv run ruff format .
```

### Performance Benchmarking
```bash
# Run performance benchmark
DISPLAY= MPLBACKEND=Agg uv run python scripts/benchmark02.py
# Expected: ~22 steps/second, ~45ms per step
```

## Validation Scenarios

**ALWAYS run these validation scenarios after making changes to ensure functionality:**

### 1. Basic Environment Creation Test
```bash
DISPLAY= MPLBACKEND=Agg uv run python -c "
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
from robot_sf.gym_env.environment_factory import make_robot_env
print('Testing environment creation...')
env = make_robot_env(debug=True)
print('Environment created successfully')
obs, _ = env.reset()
print('Environment reset successful')
print('Observation space:', env.observation_space)
print('Action space:', env.action_space)
env.close()
print('Basic validation completed successfully')
"
```

### 2. Model Loading and Prediction Test
```bash
DISPLAY= MPLBACKEND=Agg uv run python -c "
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
from robot_sf.gym_env.environment_factory import make_robot_env
from stable_baselines3 import PPO
env = make_robot_env(debug=True)
# Use newest model for best compatibility
model = PPO.load('./model/ppo_model_retrained_10m_2025-02-01.zip', env=env)
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
print('Model loading and prediction successful')
print('Action shape:', action.shape)
env.close()
"
```

### 3. Complete Simulation Test
```bash
# Run a short simulation to validate end-to-end functionality
timeout 30 DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python examples/demo_defensive.py
```

## Common Issues and Solutions

### Build Issues
- **uv not found**: Install with `pip install uv` or curl command above
- **ffmpeg missing**: Install with `sudo apt-get install -y ffmpeg`
- **Submodules empty**: Run `git submodule update --init --recursive`

### Runtime Issues
- **Import errors**: Ensure virtual environment is activated: `source .venv/bin/activate`
- **Display errors**: Use headless mode: `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy`
- **Model loading warnings**: StableBaselines3 warnings about OpenAI Gym are normal and non-blocking
- **Model compatibility**: Use newest models for best compatibility (e.g., `ppo_model_retrained_10m_2025-02-01.zip`)

### Docker Issues
- **Docker build fails**: Network connectivity issues in CI are common. Docker builds work locally but may fail in restricted environments.
- **GPU support**: See `docs/GPU_SETUP.md` for NVIDIA Docker setup

## Repository Structure

### Key Directories
- `robot_sf/` - Main source code
- `examples/` - Demo scripts showing usage patterns
- `tests/` - Main test suite
- `test_pygame/` - GUI-specific tests
- `fast-pysf/` - PySocialForce submodule for pedestrian simulation
- `scripts/` - Training and utility scripts
- `model/` - Pre-trained models (run_043.zip, ppo_model_retrained_10m_2025-02-01.zip)
- `docs/` - Documentation including refactoring guides

### Environment Creation (New Factory Pattern)
**ALWAYS use the new factory pattern for creating environments:**

```python
from robot_sf.gym_env.environment_factory import (
    make_robot_env,
    make_image_robot_env,
    make_pedestrian_env
)

# Basic robot environment
env = make_robot_env(debug=True)

# Robot with image observations
env = make_image_robot_env(debug=True)

# Pedestrian environment (requires trained robot model)
env = make_pedestrian_env(robot_model=model, debug=True)
```

### Configuration
```python
from robot_sf.gym_env.unified_config import (
    RobotSimulationConfig,
    ImageRobotConfig,
    PedestrianSimulationConfig
)

config = RobotSimulationConfig()
config.peds_have_obstacle_forces = True
env = make_robot_env(config=config)
```

## CI/CD Pipeline

The project uses GitHub Actions (`.github/workflows/ci.yml`):
- **Test job**: Runs `uv run pytest tests` 
- **Lint job**: Runs `uv run ruff check .` and `uv run ruff format --check .`

**ALWAYS run these commands before committing:**
```bash
uv run ruff check .
uv run ruff format .
uv run pytest tests
```

## Training and Examples

### Available Demos
```bash
# Visual debugging demos (require models)
uv run python examples/demo_offensive.py
uv run python examples/demo_defensive.py
uv run python examples/demo_pedestrian.py

# Refactored environment examples
uv run python examples/demo_refactored_environments.py
```

### Training Scripts
```bash
# StableBaselines PPO training
uv run python scripts/training_ppo.py

# Hyperparameter optimization
uv run python scripts/hparam_opt.py

# Evaluation
uv run python scripts/evaluate.py
```

### Docker Training (Advanced)
```bash
# Build and run GPU training (requires NVIDIA Docker)
docker compose build && docker compose run robotsf-cuda python ./scripts/training_ppo.py
# NOTE: May fail in CI environments due to network restrictions
```

## Performance Expectations

- **Environment creation**: < 1 second
- **Model loading**: 1-5 seconds
- **Simulation performance**: ~22 steps/second (~45ms per step)
- **Build time**: 2-3 minutes (first time)
- **Test suite**: 2-3 minutes (170 tests)

## Migration Notes

The project recently migrated to UV package manager and uses a new factory pattern for environment creation. See:
- `docs/UV_MIGRATION.md` - UV migration guide
- `docs/refactoring/` - Environment refactoring documentation

**Always use the new factory pattern (`make_robot_env`) instead of direct class instantiation.**

## Quick Reference Commands

```bash
# Fresh setup
git clone --recurse-submodules https://github.com/ll7/robot_sf_ll7 && cd robot_sf_ll7
sudo apt-get install -y ffmpeg && pip install uv
uv sync && source .venv/bin/activate

# Validate changes
uv run ruff check . && uv run ruff format . && uv run pytest tests

# Test functionality
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; env = make_robot_env(); env.reset(); print('OK')"

# Performance check
DISPLAY= MPLBACKEND=Agg uv run python scripts/benchmark02.py
```