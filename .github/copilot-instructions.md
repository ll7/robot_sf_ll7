# Robot SF - Robotics Simulation Environment

Robot SF is a Python-based robotics simulation environment for training robots to navigate in pedestrian-filled spaces using reinforcement learning. The project uses gymnasium (OpenAI Gym) interface with StableBaselines3 for RL training and PySocialForce for pedestrian simulation.

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## General Behavior

### Code Quality Standards

- Always follow the latest coding standards and best practices for the language being used
- Use clear, descriptive variable and function names that express intent
- Write code that is easy to read, understand, and maintain
- Ensure that all code is well-documented with meaningful comments and docstrings
- Follow the existing code style and patterns in the project
- Write comprehensive unit tests for new features and bug fixes
  - Tests should be placed in the `tests/` directory or in the `test_pygame/` directory for tests that need a display output.
- Perform code reviews and ensure changes meet quality standards
- Use the linting task and the test task to ensure code quality before committing changes

### Version Control & Collaboration

- Use version control best practices with meaningful, descriptive commit messages
- When making changes, ensure backward compatibility unless explicitly specified otherwise
- Always check for existing issues, discussions, or similar work before starting new tasks
- Use issue numbers in commit messages to link changes to specific GitHub issues
  - Format: `fix: resolve button alignment issue (#42)`
- Create feature branches named after the issue number and title in kebab-case
  - Format: `feature/42-fix-button-alignment` or `bugfix/123-memory-leak-fix`
- Keep branches up-to-date with the main branch to avoid merge conflicts
- Use pull requests for code reviews and team discussions before merging
- Always run the full test suite before merging changes to ensure system stability

#### Examples

**Branch Naming:**
```
feature/42-fix-button-alignment
bugfix/89-memory-leak-in-simulator
enhancement/156-improve-lidar-performance
```

**Commit Messages:**
```
fix: resolve 2x speed multiplier in VisualizableSimState (#42)
feat: add new lidar sensor configuration options (#156)
docs: update installation guide with GPU setup instructions
test: add comprehensive integration tests for pedestrian simulation
```

### Problem-Solving Approach

- Break down complex problems into smaller, manageable tasks
- Research existing solutions and patterns before implementing new approaches
- Consider the impact of changes on the entire system, not just the immediate problem
- Document architectural decisions and trade-offs made during implementation
- Think about edge cases, error handling, and potential failure modes

## Documentation Standards

### Technical Documentation

- Create comprehensive documentation for all significant changes and new features
- Save documentation files in the `docs/` directory using a clear folder structure
- Each major feature or issue should have its own subfolder named in kebab-case
  - Format: `docs/42-fix-button-alignment/` or `docs/feature-name/`
- Use descriptive README.md files as the main documentation entry point for each folder

### Documentation Content Requirements

Documentation should include:
- **Problem Statement**: Clear description of the issue being addressed
- **Solution Overview**: High-level approach and architectural decisions
- **Implementation Details**: Code examples, API changes, and technical specifics
- **Impact Analysis**: What systems/users are affected and how
- **Testing Strategy**: How the changes were validated
- **Future Considerations**: Potential improvements or known limitations
- **Related Links**: References to GitHub issues, pull requests, or external resources

### Documentation Best Practices

- Use proper markdown formatting with clear headings and structure
- Include code examples with syntax highlighting
- Add diagrams or screenshots when they improve understanding
  - Mermaid diagrams are welcome and encouraged for visualizing workflows, architecture, and relationships
- Write for future developers who may be unfamiliar with the context
- Keep documentation up-to-date as code evolves
- Use consistent formatting and follow markdown linting standards

## Installation and Setup

For complete installation instructions, see the [main README.md](../README.md#installation). The project uses `uv` for modern Python dependency management.

## Working Effectively

### Quick Start Commands
```bash
# Essential setup (assumes uv is installed)
git submodule update --init --recursive  # Initialize submodules
uv sync && source .venv/bin/activate     # Install dependencies and activate venv
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

# Run GUI tests in headless mode (avoids display issues)
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest test_pygame
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

# Fix formatting automatically
uv run ruff format .
```

### Performance Benchmarking (Optional)
```bash
# Run performance benchmark (only when performance impact is suspected)
DISPLAY= MPLBACKEND=Agg uv run python scripts/benchmark02.py
# Expected: ~22 steps/second, ~45ms per step
```

## Validation Scenarios

**ALWAYS run these validation scenarios after making changes to ensure functionality:**

### 1. Basic Environment Creation Test
```bash
./scripts/validation/test_basic_environment.sh
```

### 2. Model Loading and Prediction Test
```bash
./scripts/validation/test_model_prediction.sh
```

### 3. Complete Simulation Test
```bash
./scripts/validation/test_complete_simulation.sh
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

## Project-Specific Guidelines

### Robot SF Development

- This project focuses on robotic simulation and reinforcement learning
- Pay special attention to data integrity in simulation states and analysis
- Ensure consistency between simulation data generation and analysis pipelines
- Consider the impact on research workflows and data analysis tools
- Maintain compatibility with the fast-pysf reference implementation when applicable
- Test changes thoroughly as they may affect both simulation behavior and research results

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
# Setup after installation (see README.md for full installation)
git submodule update --init --recursive && uv sync && source .venv/bin/activate

# Validate changes
uv run ruff check . && uv run ruff format . && uv run pytest tests

# Test functionality
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; env = make_robot_env(); env.reset(); print('OK')"

# Performance check (optional)
DISPLAY= MPLBACKEND=Agg uv run python scripts/benchmark02.py
```