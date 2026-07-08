# Research: Ruff Linting Rules Expansion

## Overview
This research phase analyzes the proposed expansion of Ruff linting rules for the Robot SF codebase, focusing on a research-heavy Python project with ROS2/CARLA integrations.

## Selected Rule Families Analysis

### Bug Catchers & Safety (B, BLE, TRY, A, ARG, S)
- **B (flake8-bugbear)**: Catches common footguns like mutable defaults (B006/B008), misused context managers (B904)
- **BLE (flake8-blind-except)**: Prevents bare/overbroad except clauses that hide bugs
- **TRY (tryceratops)**: Detects problematic try/except patterns (TRY300/TRY203)
- **A (flake8-builtins)**: Prevents shadowing of builtin functions
- **ARG (flake8-unused-arguments)**: Flags unused parameters in public APIs
- **S (Bandit)**: Security checks for unsafe patterns (subprocess, yaml.load)

**Rationale**: Research codebases often have complex logic where these bugs can hide; safety rules are critical for reproducible simulations.

### Modernization & Simplification (UP, SIM, C4, PTH, ICN)
- **UP (pyupgrade)**: Modernizes syntax (f-strings, pathlib) with safe auto-fixes
- **SIM (flake8-simplify)**: Removes unnecessary code branches and loops
- **C4 (flake8-comprehensions)**: Cleaner list/dict/set comprehensions and generators
- **PTH (flake8-use-pathlib)**: Prefers pathlib over os.path
- **ICN (flake8-import-conventions)**: Standard import aliases (np, pd, plt)

**Rationale**: Keeps code current and readable; auto-fixes reduce manual effort in large codebases.

### Performance & Correctness (PERF, PL)
- **PERF (perflint)**: Micro-optimizations and performance traps
- **PL (pylint)**: Selected rules for code quality (excluding noisy ones)

**Rationale**: Important for simulation performance; research code needs efficiency.

### Time Handling (DTZ)
- **DTZ (flake8-datetimez)**: Timezone-aware datetime usage

**Rationale**: Simulations often involve timing; prevents subtle bugs in logging/timestamps.

### Logging & Code Quality (G, T20, ERA, COM, ISC, RUF, PGH, TCH, TID, N)
- **G (flake8-logging-format)**: Proper logging string formatting
- **T20 (flake8-print)**: Discourages print() outside scripts/tests
- **ERA (eradicate)**: Removes commented-out code
- **COM (trailing-comma)**: Consistent trailing commas
- **ISC (flake8-implicit-str-concat)**: Prevents implicit string concatenation issues
- **RUF (Ruff)**: Ruff-specific rules (unused noqa)
- **PGH (pygrep-hooks)**: Hygiene for noqa comments
- **TCH (flake8-type-checking)**: Moves typing imports under TYPE_CHECKING
- **TID (flake8-tidy-imports)**: Relative vs absolute imports
- **N (pep8-naming)**: PEP8 naming conventions

**Rationale**: Maintains code hygiene; especially important in collaborative research projects.

## Exclusions for Research Context
- **S (Security)**: Excluded globally as not critical in non-production research code
- **PLR0911/0912/0913/0915 (Pylint complexity)**: Too noisy for scientific code with complex algorithms
- **PLR2004 (Magic values)**: Common in configs and scientific constants

## Per-File Ignores
- **tests/**: Allows asserts, prints, security tests
- **scripts/**: Allows prints for CLI output
- **examples/**: Allows prints for demos
- **docs/**: Allows leniencies for documentation

## Alternatives Considered
1. **Select ["ALL"] with ignores**: More future-proof but potentially overwhelming initially
2. **Minimal expansion**: Less comprehensive bug catching
3. **Stricter rules**: Higher maintenance burden for research team

## Performance Impact Assessment
- Expected CI time: <60 seconds (target)
- Auto-fixable rules reduce manual fixes
- Research shows Ruff is 10-100x faster than flake8 equivalents

## Conclusion
This rule set provides high signal-to-noise ratio for research codebases, balancing safety, modernization, and practicality. The exclusions acknowledge research context while maintaining quality standards.