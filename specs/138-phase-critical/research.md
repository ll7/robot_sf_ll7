# Research Findings: Type Checking Fixes

## Current Type Diagnostics Analysis

**Decision**: Categorize the 103 type diagnostics from uvx ty check output
**Rationale**: Understanding the breakdown is essential for prioritizing fixes across the 4 phases
**Findings**:
- Invalid argument types: 45 issues (factory functions, data analysis)
- Unresolved imports: 23 issues (optional dependencies, dynamic imports)
- Invalid assignments: 18 issues (type mismatches, missing annotations)
- Missing arguments: 17 issues (factory function calls)

**Alternatives considered**: Manual counting vs automated parsing - chose manual analysis for accuracy

## Python 3.11+ datetime.UTC Compatibility

**Decision**: Use `datetime.timezone.utc` for Python 3.8-3.10 compatibility, direct `datetime.UTC` for 3.11+
**Rationale**: datetime.UTC was introduced in Python 3.11, requiring compatibility imports for older versions
**Implementation**: Conditional import pattern with try/except for backward compatibility

**Alternatives considered**: Always use timezone.utc vs conditional imports - chose conditional for optimal performance

## Factory Function Type Annotations

**Decision**: Update return type annotations to use `gymnasium.Env` generic types
**Rationale**: Provides accurate type information for IDE support and static analysis
**Pattern**: `def make_robot_env(...) -> gymnasium.Env[ObsType, ActType]:`

**Alternatives considered**: Use `Any` vs specific generic types - chose generics for type safety

## Gym Space Type Issues

**Decision**: Use `gymnasium.spaces.Space` base class for type annotations
**Rationale**: Provides proper typing for reinforcement learning spaces while maintaining compatibility
**Pattern**: `observation_space: gymnasium.spaces.Space`

**Alternatives considered**: Specific space types vs base class - chose base class for flexibility

## Optional Dependencies Handling

**Decision**: Keep dependencies optional with conditional imports using TYPE_CHECKING
**Rationale**: Maintains lightweight installation while providing type safety when dependencies are present
**Pattern**: `if TYPE_CHECKING: import optional_dep`

**Alternatives considered**: Make all dependencies required vs optional - chose optional to preserve current architecture

## uvx ty Configuration

**Decision**: Use default uvx ty settings with --exit-zero for CI compatibility
**Rationale**: Standard mypy-based checking provides comprehensive type analysis
**Configuration**: No custom config file needed, relies on pyproject.toml settings

**Alternatives considered**: Custom mypy config vs default - chose default for simplicity