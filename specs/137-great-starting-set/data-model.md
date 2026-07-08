# Data Model: Ruff Linting Configuration

## Overview
The data model for this feature is the Ruff linting configuration stored in `pyproject.toml` under the `[tool.ruff.lint]` section. This configuration controls code quality rules, ignores, and per-file exceptions.

## Configuration Structure

### [tool.ruff.lint]
Root section for Ruff linting settings.

#### select
Array of rule codes to enable. Each code represents a rule family:
- **E4/E7/E9/F/W/C901/I001**: Base rules (syntax errors, pyflakes, warnings, complexity, import sorting)
- **B/BLE/TRY/A/ARG/S**: Bug catchers and safety
- **UP/SIM/C4/PTH/ICN**: Modernization and simplification
- **PERF/PL**: Performance and quality
- **DTZ**: Time handling
- **G/T20/ERA/COM/ISC**: Logging and style
- **RUF/PGH/TCH/TID/N**: Housekeeping

#### ignore
Array of rule codes to disable globally:
- **PLR0911/0912/0913/0915**: Pylint complexity rules (too noisy for research code)
- **PLR2004**: Magic value comparisons (common in scientific code)
- **S**: Security checks (not critical in research context)

#### per-file-ignores
Dictionary mapping glob patterns to arrays of ignored rules:
- **"tests/**/*"**: [S101, T201, PLR2004] - Allow asserts, prints, magic values in tests
- **"scripts/**/*"**: [T201] - Allow prints in scripts
- **"examples/**/*"**: [T201] - Allow prints in examples
- **"docs/**/*"**: [T201] - Allow prints in docs

## Schema Validation
The configuration must conform to Ruff's TOML schema. Invalid rule codes or malformed globs will cause Ruff to error.

## Relationships
- **pyproject.toml**: Contains the configuration
- **Source files**: Matched by glob patterns in per-file-ignores
- **CI pipeline**: Uses this configuration for linting checks
- **Development workflow**: Developers run `ruff check --fix` locally

## Constraints
- Rule codes must be valid Ruff rule identifiers
- Glob patterns must be valid for file matching
- Configuration must not break existing CI (performance <60s, no new failures)
- Per-file ignores should be minimal and justified

## Evolution
Future updates may add new rules or adjust ignores based on codebase growth and team feedback.