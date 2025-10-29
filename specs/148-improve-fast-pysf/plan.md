# Implementation Plan: Improve fast-pysf Integration

**Branch**: `148-improve-fast-pysf` | **Date**: 2025-10-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/148-improve-fast-pysf/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Integrate fast-pysf physics tests into the main pytest suite, resolve all 24 PR #236 review comments, extend quality tooling (ruff, ty, coverage) to fast-pysf/, and improve type annotations while maintaining functional correctness. This ensures the fast-pysf subtree meets the same quality standards as robot_sf core code and enables unified testing, linting, and type checking workflows.

## Technical Context

**Language/Version**: Python 3.13.1 (project standard, already in use)  
**Primary Dependencies**: 
- pytest 8.3.4 (test framework)
- ruff 0.9.2+ (linting/formatting)
- ty (type checking via uvx)
- uv (package manager and task runner)
- fast-pysf subtree (NumPy 1.26.4, numba 0.60.0 for JIT-compiled physics)

**Storage**: N/A (test fixtures use file-based map JSON in fast-pysf/tests/test_maps/)  
**Testing**: pytest with coverage plugin, headless mode via environment variables (DISPLAY=, MPLBACKEND=Agg, SDL_VIDEODRIVER=dummy)  
**Target Platform**: macOS (dev), Linux (CI), headless execution required  
**Project Type**: Single Python library with integrated subtree (fast-pysf/ as git subtree)  
**Performance Goals**: 
- Test suite execution < 5 minutes total (robot_sf + fast-pysf combined)
- Fast-pysf tests < 60 seconds
- No degradation to existing ~43 robot_sf tests performance

**Constraints**: 
- Must maintain backward compatibility with FastPysfWrapper integration point
- Type annotations cannot break numba JIT compilation (@njit decorated functions)
- Ruff and ty configurations must not conflict with existing robot_sf settings
- All tests must run headless for CI compatibility

**Scale/Scope**: 
- Current: ~43 tests in tests/, 12 tests in fast-pysf/tests/ (2 failing due to missing fixtures)
- Target: 55+ unified tests, all passing
- PR #236 review comments: 24 issues (7 high, 10 medium, 7 low priority)
- Estimated effort: 18-25 hours across 4 phases

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Compliance Analysis

**I. Reproducible Social Navigation Research Core**: ✅ PASS
- Feature maintains reproducibility by unifying test execution
- No changes to deterministic simulation or benchmark outputs
- Test infrastructure improvements support reproducibility validation

**II. Factory-Based Environment Abstraction**: ✅ PASS  
- No changes to factory functions or environment interfaces
- FastPysfWrapper integration point preserved
- Quality improvements are internal to fast-pysf subtree

**III. Benchmark & Metrics First**: ✅ PASS
- No benchmark schema changes
- PR review resolution includes test result verification (high priority issue)
- Maintains structured episode output contracts

**IV. Unified Configuration & Deterministic Seeds**: ✅ PASS
- No configuration schema changes
- Test infrastructure maintains seed propagation patterns
- Quality tooling activation doesn't affect runtime determinism

**V. Minimal, Documented Baselines**: ✅ PASS
- No baseline algorithm changes
- Documentation updates align with constitution requirement
- Test integration validates baseline compatibility

**VI. Metrics Transparency & Statistical Rigor**: ✅ PASS
- No metrics computation changes
- Test infrastructure supports metrics validation
- Coverage reporting enhances transparency

**VII. Backward Compatibility & Evolution Gates**: ✅ PASS
- No breaking changes to public APIs
- Internal quality improvements only
- Test integration is additive (doesn't break existing tests)

**VIII. Documentation as an API Surface**: ✅ PASS
- Documentation updates planned (Phase 1, Task 1.3)
- dev_guide.md and README.md to be updated
- All changes will be documented in appropriate sections

**IX. Test Coverage for Public Behavior**: ✅ PASS (ENHANCING)
- Feature explicitly adds test coverage for fast-pysf
- Target: coverage ≥70% for fast-pysf code
- Integration tests validate unified test suite

**X. Scope Discipline**: ✅ PASS
- Strictly internal quality and testing improvements
- No scope expansion beyond social navigation simulation
- Fast-pysf remains focused on pedestrian physics

**XI. Library Reuse & Helper Documentation**: ✅ PASS
- PR review resolution includes documentation improvements (medium priority: placeholders, copy-paste docstrings)
- Type annotation improvements will enhance API clarity
- Helper documentation maintained in line with constitution

**XII. Preferred Logging & Observability**: ✅ PASS
- No logging framework changes in this feature
- Fast-pysf subtree uses print statements (separate codebase, acceptable per constitution)
- Robot_sf integration layer (FastPysfWrapper) already uses proper logging

### Overall Gate Status: ✅ ALL GATES PASS

No constitution violations. Feature is scoped as internal quality improvements that strengthen compliance with Principles VIII (Documentation), IX (Test Coverage), and XI (Helper Documentation).

## Project Structure

## Project Structure

### Documentation (this feature)

```
specs/148-improve-fast-pysf/
├── plan.md              # This file (/speckit.plan command output)
├── spec.md              # Feature specification (already created)
├── implementation_plan.md  # Detailed 4-phase plan (already created)
├── pr236_review_comments.md  # PR review analysis (already created)
├── README.md            # Executive summary (already created)
├── checklists/
│   └── requirements.md  # Quality validation (already created)
├── research.md          # Phase 0 output (to be created)
├── data-model.md        # Phase 1 output (to be created)
├── quickstart.md        # Phase 1 output (to be created)
├── contracts/           # Phase 1 output (to be created)
└── tasks.md             # Phase 2 output (/speckit.tasks - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
robot_sf_ll7/
├── robot_sf/           # Main library (quality standards already applied)
│   ├── gym_env/       # Factory functions and environments
│   ├── sim/           # FastPysfWrapper integration point
│   ├── benchmark/     # Runner and metrics
│   └── ...
├── fast-pysf/         # Git subtree (TARGET for quality improvements)
│   ├── pysocialforce/ # Physics engine implementation
│   │   ├── forces.py  # Force computations (PR review issues)
│   │   ├── map_config.py  # Map configuration (PR review issues)
│   │   ├── map_loader.py  # Map loading (failing tests)
│   │   ├── map_loader_svg.py  # SVG map support (PR review issues)
│   │   ├── map_osm_converter.py  # OSM conversion (PR review issues)
│   │   ├── scene.py   # Scene management (PR review issues)
│   │   └── simulator.py  # Main simulator (PR review issues)
│   ├── tests/         # Test suite to be integrated
│   │   ├── test_forces.py  # Force tests (3 pass)
│   │   ├── test_map_loader.py  # Map loader tests (2 fail - missing fixtures)
│   │   ├── test_simulator.py  # Simulator tests (2 pass)
│   │   ├── unittest/
│   │   │   └── test_forces.py  # Centroid tests (5 pass)
│   │   └── test_maps/  # Test fixtures (MISSING - to be created/fixed)
│   ├── examples/      # Demo scripts (PR review issues)
│   ├── README.md      # Subtree documentation (PR review issues)
│   └── .github/       # CI configuration (PR review issues)
├── tests/             # Main test suite (~43 tests, all passing)
├── test_pygame/       # GUI tests (separate suite)
├── pyproject.toml     # Configuration (to be updated for pytest/ruff/coverage)
├── .github/
│   └── workflows/
│       └── ci.yml     # CI pipeline (PR review issues)
└── docs/
    ├── dev_guide.md   # To be updated with test integration info
    └── README.md      # To be updated with new test commands
```

**Structure Decision**: Single Python library with integrated subtree pattern. The fast-pysf/ directory is a git subtree (not submodule) containing the pedestrian physics engine. Quality improvements target fast-pysf/ while maintaining integration through robot_sf/sim/FastPysfWrapper.py. Test integration adds fast-pysf/tests/ to pytest discovery without restructuring existing layout.

## Complexity Tracking

*This feature has NO constitution violations. Table left for reference.*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |

**Justification**: All improvements are internal quality enhancements that strengthen existing constitution compliance (Principles VIII, IX, XI). No additional complexity introduced.

---

## Phase 1 Completion Status

✅ **Phase 0 (Research)**: Completed
- All technical unknowns resolved
- Decisions documented in `research.md`
- 6 research areas covered (pytest, ruff, ty, coverage, fixtures, PR comments)

✅ **Phase 1 (Design & Contracts)**: Completed
- Data model defined in `data-model.md` (7 entities with relationships)
- API contracts documented in `contracts/README.md` (7 contract types)
- Quickstart guide created in `quickstart.md`
- Agent context updated (copilot instructions)

✅ **Constitution Check Re-evaluation**: All gates PASS (unchanged from initial check)

### Post-Design Constitution Verification

**No new violations introduced**. The design phase reinforced compliance:

1. **Principle VIII (Documentation as API Surface)**: ✅ ENHANCED
   - Created comprehensive quickstart.md for developer onboarding
   - Documented all configuration contracts in contracts/README.md
   - Clear API surface definition for test execution and quality checks

2. **Principle IX (Test Coverage for Public Behavior)**: ✅ ENHANCED
   - Data model includes TestSuite and TestCase entities with coverage tracking
   - Coverage contract specifies ≥70% target for fast-pysf
   - Validation rules ensure no regression in robot_sf coverage

3. **Principle XI (Library Reuse & Helper Documentation)**: ✅ ENHANCED
   - PR review resolution includes documentation improvements
   - Type annotation strategy preserves helper clarity
   - Data model enforces documentation of resolution notes

4. **All Other Principles**: ✅ MAINTAINED
   - No changes to environment factories (II)
   - No benchmark schema changes (III)
   - No configuration contract breaking changes (IV, VII)
   - No scope expansion (X)
   - Logging requirements respected (XII)

**Conclusion**: Design phase successfully elaborated implementation details without introducing technical debt or constitution violations. Feature remains scoped as internal quality improvements with strengthened documentation and testing infrastructure.

---

## Implementation Readiness

**Status**: ✅ Ready for Phase 2 (/speckit.tasks)

**Deliverables Completed**:
- [x] plan.md (this file)
- [x] research.md (6 research areas, all NEEDS CLARIFICATION resolved)
- [x] data-model.md (7 entities, relationships, validation rules)
- [x] contracts/README.md (7 API contracts with schemas and examples)
- [x] quickstart.md (developer onboarding guide)
- [x] Agent context updated (.github/copilot-instructions.md)
- [x] Constitution Check re-evaluated (all gates pass)

**Next Phase**: Use `/speckit.tasks` to generate granular task breakdown from this plan.

**Estimated Implementation Timeline**:
- **Phase 1 (Test Integration)**: 4 hours
- **Phase 2 (PR Review Resolution)**: 7-11 hours
- **Phase 3 (Quality Tooling)**: 3-4 hours
- **Phase 4 (Type Annotations)**: 4-6 hours
- **Total**: 18-25 hours (3-4 working days)

---

## References

- Feature specification: [spec.md](./spec.md)
- Existing implementation plan: [implementation_plan.md](./implementation_plan.md)
- PR review comments analysis: [pr236_review_comments.md](./pr236_review_comments.md)
- Requirements checklist: [checklists/requirements.md](./checklists/requirements.md)
- Constitution: [.specify/memory/constitution.md](../../.specify/memory/constitution.md)
- Development guide: [docs/dev_guide.md](../../docs/dev_guide.md)
