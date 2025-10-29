# Type Annotation Plan: fast-pysf

**Created**: 2025-10-29  
**Phase**: User Story 4 - Improved Type Annotations (P3)  
**Goal**: Reduce type errors by ≥25% while maintaining numba JIT compatibility

## Baseline Analysis (T055)

### Current State
- **Total type check output lines**: 275
- **Total errors/warnings**: 18 (17 errors, 1 warning)
- **Files with issues**: 
  - `sim_view.py` (majority of errors)
  - `forces.py` (warnings about missing attributes)
  - Other files (minor issues)

### Error Breakdown by Category

**Top Issues Identified**:
1. **invalid-type-form** - Using `np.array` as type annotation
2. **invalid-assignment** - Type mismatches in assignments
3. **invalid-return-type** - Return type doesn't match declared type
4. **no-matching-overload** - Incorrect function call signatures
5. **invalid-argument-type** - Wrong types passed to functions
6. **possibly-missing-attribute** - Attributes may not exist on objects

### Files Prioritized for Annotation

Based on error analysis and public API importance:

1. **forces.py** (Priority: HIGH)
   - Public API: `desired_force`, `social_force`, `obstacle_force`, `ortho_vec`
   - Issue: `possibly-missing-attribute` warnings
   - Numba compatibility: CRITICAL (many `@njit` decorated functions)

2. **simulator.py** (Priority: HIGH)
   - Public API: `Simulator`, `Simulator_v2` classes
   - Issue: Some type annotations exist but incomplete
   - Numba compatibility: Some methods use numba

3. **map_loader.py** (Priority: MEDIUM)
   - Public API: `load_map` function
   - Issue: Missing annotations for file paths
   - Numba compatibility: None

4. **scene.py** (Priority: MEDIUM)
   - Public API: `PedState`, `EnvState` classes
   - Issue: Missing annotations for state arrays
   - Numba compatibility: Some methods

5. **sim_view.py** (Priority: LOW - not library code)
   - Visualization only, not part of core API
   - Has most errors but less critical for library users

## Annotation Strategy

### Principles

1. **Numba Compatibility First**
   - Do NOT add type hints to `@njit` decorated function internals
   - Only annotate function signatures (parameters and return types)
   - Test numba compilation after each change

2. **Public API Priority**
   - Focus on externally-facing functions and classes
   - Internal helpers can remain untyped for now

3. **NumPy Array Types**
   - Use `npt.NDArray[np.float64]` for typed arrays
   - Use `np.ndarray` for generic arrays
   - Import from `numpy.typing as npt`

4. **Gradual Typing**
   - Start with simple function signatures
   - Use `Any` temporarily where complex inference needed
   - Document why `Any` is used

### Tasks Breakdown

#### T057: forces.py Type Hints
**Target Functions**:
- `desired_force()`
- `social_force()`  
- `obstacle_force()`
- `ortho_vec()`

**Approach**:
- Add parameter types (arrays, config objects)
- Add return type annotations
- Keep `@njit` internals untyped
- Use `npt.NDArray[np.float64]` for arrays

**Numba Risk**: MEDIUM (test after each change)

#### T058: simulator.py Type Hints
**Target Classes/Methods**:
- `Simulator.__init__()`
- `Simulator.step()`
- `Simulator.step_once()`
- `Simulator_v2` equivalents

**Approach**:
- Annotate init parameters
- Type `peds`, `groups`, `obstacles` collections
- Return types for step methods

**Numba Risk**: LOW (main orchestration code)

#### T059: map_loader.py Type Hints
**Target Functions**:
- `load_map(file_path: str | Path) -> MapDefinition`
- Helper functions

**Approach**:
- Use `str | Path` for file paths
- Return `MapDefinition` explicitly
- Type dict parsing helpers

**Numba Risk**: NONE

#### T060: scene.py Type Hints
**Target Classes**:
- `PedState`
- `EnvState`
- Scene methods

**Approach**:
- Annotate state array types
- Type list/tuple parameters
- Method return types

**Numba Risk**: LOW

### Success Metrics

**Quantitative**:
- Baseline errors: 18
- Target errors: ≤13 (25% reduction minimum)
- Ideal: ≤9 (50% reduction)

**Qualitative**:
- All public API functions typed
- No numba compilation failures
- No test failures
- All `Any` usage documented

## Implementation Order

1. **Setup** (5 min)
   - Add numpy.typing imports where needed
   - Verify current tests pass

2. **Quick Wins** (30 min)
   - T059: map_loader.py (easiest, no numba)
   - Verify type check improvement

3. **Core APIs** (1-2 hours)
   - T057: forces.py public functions
   - T058: simulator.py main classes
   - Test numba compatibility after each

4. **Secondary** (30 min)
   - T060: scene.py classes
   - Clean up remaining simple issues

5. **Verification** (30 min)
   - T062: Run type check, compare counts
   - T063: Verify numba tests pass
   - Document remaining issues

## Risks and Mitigations

### Risk 1: Numba Incompatibility
**Mitigation**: 
- Only type function signatures, not internals
- Test after each change: `uv run pytest fast-pysf/tests/test_forces.py -v`
- Keep backup of working version

### Risk 2: Complex Generic Types
**Mitigation**:
- Use `Any` temporarily with TODO comment
- Document in T064 for future improvement

### Risk 3: Runtime Performance
**Mitigation**:
- Type hints don't affect runtime in CPython
- Numba ignores type hints (uses its own inference)

## Notes

- Type annotations are for static analysis only
- Numba uses its own type inference system
- Focus on developer experience (IDE autocomplete, type checking)
- Don't over-engineer - simple types are better than complex generics

## References

- NumPy typing: https://numpy.org/devdocs/reference/typing.html
- Numba compatibility: https://numba.pydata.org/numba-doc/latest/reference/pysupported.html
- Python typing docs: https://docs.python.org/3/library/typing.html
