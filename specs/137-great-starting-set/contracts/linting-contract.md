# Linting Contract

## Purpose
This contract defines the behavioral guarantees and quality standards for the expanded Ruff linting configuration in the Robot SF codebase.

## Functional Contracts

### Rule Application Contract
- **Precondition**: Valid Python source files in the repository
- **Action**: Run `ruff check --fix .` on the codebase
- **Postcondition**: 
  - No syntax errors remain
  - Auto-fixable style issues are corrected
  - Non-fixable issues are flagged with clear error messages
  - CI linting completes in under 60 seconds
  - No increase in CI build failures

### Configuration Stability Contract
- **Precondition**: pyproject.toml contains valid Ruff configuration
- **Invariant**: Configuration remains backward compatible
- **Postcondition**: Future Ruff versions can use the configuration without breaking changes

### Per-File Ignore Contract
- **Precondition**: Source files match defined glob patterns
- **Action**: Apply per-file ignores during linting
- **Postcondition**: 
  - Tests can use assert statements without warnings
  - Scripts can use print statements without warnings
  - Examples and docs have appropriate leniencies
  - Library code maintains strict standards

## Quality Contracts

### Bug Prevention Contract
- **Guarantee**: Enabled rules catch common Python bugs (mutable defaults, bare except, builtin shadowing)
- **Evidence**: Rules B006, B008, BLE001, A001, TRY003 are active
- **Validation**: Code review and testing catch these issues

### Code Modernization Contract
- **Guarantee**: Code uses modern Python patterns (f-strings, pathlib, comprehensions)
- **Evidence**: Rules UP, PTH, C4, SIM are active
- **Validation**: Auto-fixes modernize existing code

### Performance Contract
- **Guarantee**: Linting does not significantly impact CI performance
- **Evidence**: Target <60 seconds execution time
- **Validation**: CI timing measurements

### Research Code Compatibility Contract
- **Guarantee**: Rules accommodate scientific computing patterns
- **Evidence**: Exclusions for complexity rules and magic values
- **Validation**: No false positives in research code

## Error Handling Contracts

### Configuration Error Contract
- **Precondition**: Invalid rule code or malformed glob in pyproject.toml
- **Action**: Run Ruff
- **Postcondition**: Clear error message indicating the problem
- **Recovery**: Fix configuration and re-run

### Unfixable Issue Contract
- **Precondition**: Code violates a non-auto-fixable rule
- **Action**: Run Ruff
- **Postcondition**: Descriptive error message with rule code and location
- **Recovery**: Manual code fix required

## Monitoring Contracts

### CI Integration Contract
- **Precondition**: Code changes committed
- **Action**: CI runs linting
- **Postcondition**: 
  - Pass/fail status reported
  - Execution time logged
  - No regressions in failure count

### Local Development Contract
- **Precondition**: Developer runs `ruff check --fix`
- **Action**: Linting executes
- **Postcondition**: 
  - Fixes applied automatically where possible
  - Remaining issues reported clearly
  - No data loss or corruption