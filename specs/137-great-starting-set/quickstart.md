# Quickstart: Expanded Ruff Linting Rules

## Overview
This guide shows how to apply and use the expanded Ruff linting configuration for improved code quality in the Robot SF project.

## Prerequisites
- Python environment with Ruff installed
- Access to pyproject.toml in repository root

## Configuration Update

### 1. Update pyproject.toml
Add the following section to your `pyproject.toml`:

```toml
[tool.ruff.lint]
select = [
  # Base rules
  "E4","E7","E9","F","W","C901","I001",
  # Bug catchers & safety
  "B","BLE","TRY","A","ARG","S",
  # Modernization & simplification
  "UP","SIM","C4","PTH","ICN",
  # Performance & correctness
  "PERF","PL",
  # Time handling
  "DTZ",
  # Logging & code quality
  "G","T20","ERA","COM","ISC",
  # Housekeeping
  "RUF","PGH","TCH","TID","N",
]

ignore = [
  "PLR0911","PLR0912","PLR0913","PLR0915",  # Complexity rules
  "PLR2004",  # Magic values
  "S",  # Security checks
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*" = ["S101","T201","PLR2004"]
"scripts/**/*" = ["T201"]
"examples/**/*" = ["T201"]
"docs/**/*" = ["T201"]
```

### 2. Install/Upgrade Ruff
```bash
uv add --dev ruff
# or
pip install ruff
```

## Usage

### Local Development

#### Check Code
```bash
# Check for issues
ruff check .

# Check and auto-fix what can be fixed
ruff check --fix .

# Check specific files
ruff check robot_sf/gym_env/
```

#### Format Code
```bash
# Format files
ruff format .

# Check formatting without changing files
ruff format --check .
```

### CI Integration
The configuration is designed for CI pipelines:

```yaml
# .github/workflows/ci.yml
- name: Lint with Ruff
  run: |
    ruff check --fix .
    ruff format --check .
  timeout-minutes: 1
```

## Common Scenarios

### Fixing Issues
```bash
# See what would be fixed
ruff check --fix --dry-run .

# Fix everything automatically
ruff check --fix .
```

### Handling Conflicts
- **Auto-fixable**: Let Ruff fix them
- **Not auto-fixable**: Manually edit code per error message
- **False positives**: Add targeted ignores if justified

### Research Code Patterns
- **Magic numbers in configs**: Allowed (PLR2004 ignored)
- **Complex algorithms**: Complexity rules ignored
- **Print statements**: Allowed in scripts/examples/tests

## Troubleshooting

### Performance Issues
- If linting takes >60 seconds, review file count or add excludes
- Use `ruff check --exclude` for large generated files

### Configuration Errors
- Invalid rule codes cause immediate failure
- Check Ruff documentation for current rule names

### CI Failures
- New rules may flag existing code
- Use `ruff check --fix` locally first
- Review and fix remaining issues

## Benefits
- **Bug Prevention**: Catches common Python mistakes
- **Code Modernization**: Updates to current Python patterns
- **Consistency**: Standardized formatting and style
- **Performance**: Fast linting with auto-fixes
- **Research-Friendly**: Accommodates scientific computing patterns

## Next Steps
- Run on existing codebase to identify issues
- Gradually fix flagged problems
- Monitor CI performance
- Adjust ignores based on team feedback