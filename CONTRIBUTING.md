# Contributing to robot_sf

Thank you for your interest in contributing to robot_sf! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that we expect all contributors to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.11+ (Python 3.12 recommended)
- `uv` package manager
- Git with submodule support
- FFmpeg (for video recording)

### Initial Setup

1. **Fork and Clone**
   ```bash
   git clone --recurse-submodules https://github.com/YOUR_USERNAME/robot_sf_ll7
   cd robot_sf_ll7
   ```

2. **Install Dependencies**
   ```bash
   # Install uv if not already installed
   pip install uv
   
   # Sync all dependencies
   uv sync --all-extras
   
   # Activate virtual environment
   source .venv/bin/activate
   ```

3. **Install Pre-commit Hooks**
   ```bash
   uv run pre-commit install
   ```

4. **Verify Installation**
   ```bash
   uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; print('Import successful')"
   ```

### Important: Submodule Initialization

This project uses the `fast-pysf` submodule for pedestrian simulation. Always ensure submodules are initialized:

```bash
git submodule update --init --recursive
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

Follow the [Development Guide](docs/dev_guide.md) for detailed development practices.

### 3. Quality Gates

Before committing, run these quality checks:

```bash
# Format and lint
uv run ruff check --fix .
uv run ruff format .

# Type checking
uvx ty check . --exit-zero

# Run tests
uv run pytest tests

# Check code quality
uv run pylint robot_sf --errors-only
```

**One-liner for all checks:**
```bash
uv run ruff check --fix . && uv run ruff format . && uv run pylint robot_sf --errors-only && uvx ty check . --exit-zero && uv run pytest tests
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add new navigation algorithm"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions or changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

## Coding Standards

### Style Guide

- **Line length**: 100 characters maximum
- **Formatting**: Use `ruff format` (Black-compatible)
- **Linting**: Must pass `ruff check`
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Required for all modules, classes, and public functions

### Docstring Format

```python
def example_function(param1: str, param2: int) -> bool:
    """Short description of the function.
    
    Longer description if needed, explaining the purpose and usage.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When input is invalid.
    """
    pass
```

### Logging

Use Loguru for logging, not `print()` statements:

```python
from loguru import logger

# Good
logger.info("Processing episode seed={seed}", seed=seed)
logger.warning("No frames recorded")

# Avoid in library code
print("Processing...")  # Only acceptable in scripts/ or examples/
```

### Architecture Patterns

- **Environment Creation**: Always use factory functions from `robot_sf.gym_env.environment_factory`
- **Configuration**: Use classes from `robot_sf.gym_env.unified_config`
- **No Direct Imports**: Don't import from `fast-pysf` directly; use `FastPysfWrapper`

## Testing Requirements

### Test Coverage

- All new features must include tests
- Aim for >80% code coverage for new code
- Tests must pass before PR approval

### Test Types

1. **Unit Tests** (`tests/`)
   ```bash
   uv run pytest tests
   ```

2. **GUI Tests** (`test_pygame/`) - Run headless:
   ```bash
   DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest test_pygame
   ```

3. **Fast-pysf Tests** (physics submodule):
   ```bash
   uv run python -m pytest fast-pysf/tests/ -v
   ```

### Writing Tests

```python
def test_example_feature():
    """Test that the feature works as expected."""
    # Arrange
    env = make_robot_env(debug=True)
    
    # Act
    obs, info = env.reset()
    
    # Assert
    assert obs is not None
    assert 'scenario_id' in info
```

### Performance Tests

If your changes affect performance, include benchmark results:

```bash
DISPLAY= MPLBACKEND=Agg uv run python scripts/benchmark02.py
```

Expected performance:
- Environment creation: < 1 second
- Simulation: ~22 steps/second (~45ms/step)

## Documentation

### Required Documentation

All significant changes require documentation:

1. **Code Documentation**: Docstrings for all public APIs
2. **User Documentation**: Update relevant files in `docs/`
3. **Design Documents**: For complex features, create `docs/dev/issues/<feature-name>/design.md`
4. **CHANGELOG**: Update `CHANGELOG.md` for user-facing changes

### Documentation Structure

```
docs/
├── README.md              # Documentation index
├── dev_guide.md          # Primary development reference
├── dev/
│   └── issues/
│       └── <feature-name>/
│           ├── design.md  # Design document
│           └── todo.md    # Task tracking
└── <topic>.md            # Topic-specific guides
```

### Documentation Best Practices

- Use clear headings and structure
- Include code examples with syntax highlighting
- Add diagrams for complex concepts (Mermaid encouraged)
- Link to related documentation
- Keep documentation up-to-date with code changes

## Pull Request Process

### Before Submitting

1. ✅ All quality gates pass locally
2. ✅ Tests added for new functionality
3. ✅ Documentation updated
4. ✅ CHANGELOG.md updated (for user-facing changes)
5. ✅ Commit messages follow conventions
6. ✅ Branch is up-to-date with main

### PR Template

Use the provided PR template and fill in all sections:

- **Description**: Clear explanation of changes
- **Motivation**: Why this change is needed
- **Testing**: How changes were tested
- **Breaking Changes**: Note any breaking changes
- **Related Issues**: Link to related issues

### Review Process

1. Automated CI checks must pass
2. At least one maintainer approval required
3. All review comments must be addressed
4. No merge conflicts with main branch

### CI Pipeline

The CI pipeline runs:
- Linting (Ruff)
- Type checking (ty)
- Unit tests (pytest)
- Coverage comparison
- Validation smoke tests
- Benchmark smoke test (main branch only)

## Issue Reporting

### Bug Reports

Use the bug report template and include:

- Clear description of the bug
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (Python version, OS, etc.)
- Relevant logs or error messages

### Feature Requests

Use the feature request template and include:

- Clear description of the feature
- Use case and motivation
- Proposed implementation (if any)
- Potential alternatives considered

### Issue Labels

We use labels to categorize issues:
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `performance`: Performance-related issues

## Community

### Communication Channels

- **Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions and discussions
- **Discussions**: For questions and general discussions (if enabled)

### Getting Help

If you need help:

1. Check the [Development Guide](docs/dev_guide.md)
2. Search existing issues and discussions
3. Review documentation in `docs/`
4. Open a new issue with the `question` label

### Recognition

Contributors are recognized in:
- Git commit history
- Release notes
- Special recognition for significant contributions

## Development Tips

### VS Code Tasks

The repository includes VS Code tasks for common operations:

- **Install Dependencies**: `uv sync`
- **Ruff: Format and Fix**: Lint and format code
- **Check Code Quality**: Run Ruff + Pylint
- **Type Check**: Run ty type checker
- **Run Tests**: Execute test suite

Access via: `Cmd/Ctrl + Shift + P` → "Tasks: Run Task"

### Debugging

For debugging environments:

```python
from robot_sf.gym_env.environment_factory import make_robot_env

env = make_robot_env(debug=True)  # Enable debug mode
obs, info = env.reset()
```

### Common Issues

**Submodule not initialized:**
```bash
git submodule update --init --recursive
```

**Import errors:**
```bash
source .venv/bin/activate
```

**Display errors (headless):**
```bash
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg
```

## License

By contributing, you agree that your contributions will be licensed under the same [GPL-3.0 License](LICENSE) that covers the project.

## Questions?

If you have questions about contributing, please:
1. Review this guide and the [Development Guide](docs/dev_guide.md)
2. Search existing issues
3. Open a new issue with your question

Thank you for contributing to robot_sf! 🤖
