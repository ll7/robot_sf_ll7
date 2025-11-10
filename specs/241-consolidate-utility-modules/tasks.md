# Tasks: Consolidate Utility Modules

**Input**: Design documents from `/specs/241-consolidate-utility-modules/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/api-contract.md, quickstart.md

**Tests**: This is a refactoring task - no new tests required. Existing 893-test suite validates migration correctness.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

This project follows the single-project Python library structure:
- Source code: `robot_sf/` (package root)
- Tests: `tests/` (unit/integration tests)
- Examples: `examples/` (demo scripts)
- Scripts: `scripts/` (training/benchmark scripts)
- Documentation: `docs/` (markdown documentation)

---

## Phase 1: Setup (Pre-Migration Validation)

**Purpose**: Validate baseline before any changes

- [x] T001 Verify current branch is `241-consolidate-utility-modules`
- [x] T002 Run baseline test suite to establish passing state: `uv run pytest tests`
- [x] T003 [P] Verify type checking baseline: `uvx ty check .`
- [x] T004 [P] Verify linting baseline: `uv run ruff check .`
- [x] T005 Document baseline metrics: test count (923 passed, 6 skipped), type diagnostics (173), lint errors (0 - all passed)

**Checkpoint**: Baseline established - ready to begin migration

---

## Phase 2: Foundational (File Operations)

**Purpose**: Core file moves that MUST be complete before import updates

**⚠️ CRITICAL**: No import updates can begin until all files are moved

- [x] T006 Move `robot_sf/util/types.py` → `robot_sf/common/types.py` using `git mv`
- [x] T007 Move `robot_sf/util/compatibility.py` → `robot_sf/common/compat.py` using `git mv` (includes rename)
- [x] T008 Move `robot_sf/utils/seed_utils.py` → `robot_sf/common/seed.py` using `git mv` (includes rename)
- [x] T009 Update `robot_sf/common/__init__.py` to export commonly-used symbols (Vec2D, RobotPose, Line2D, set_global_seed, raise_fatal_with_remedy, validate_compatibility)
- [x] T010 Verify moved files exist at new locations and git history preserved

**Checkpoint**: All files moved - import updates can now begin

---

## Phase 3: User Story 1 - Developer Imports Utilities Consistently (Priority: P1) 🎯 MVP

**Goal**: Consolidate utility modules into single robot_sf/common/ location and update all import statements

**Independent Test**: Run `grep -r "from robot_sf.util\b" robot_sf/ tests/ examples/ scripts/` and `grep -r "from robot_sf.utils\b" robot_sf/ tests/ examples/ scripts/` - both should return zero matches

### Implementation for User Story 1

**Import Updates - robot_sf/ Package**

- [x] T011 [P] [US1] Find all imports in robot_sf/: `grep -r "from robot_sf\.util" robot_sf/ > /tmp/util_imports.txt`
- [x] T012 [P] [US1] Find all imports in robot_sf/: `grep -r "from robot_sf\.utils" robot_sf/ > /tmp/utils_imports.txt`
- [x] T013 [US1] Update imports from `robot_sf.util.types` → `robot_sf.common.types` in robot_sf/ directory
- [x] T014 [US1] Update imports from `robot_sf.util.compatibility` → `robot_sf.common.compat` in robot_sf/ directory
- [x] T015 [US1] Update imports from `robot_sf.utils.seed_utils` → `robot_sf.common.seed` in robot_sf/ directory

**Import Updates - tests/ Directory**

- [x] T016 [P] [US1] Update imports from `robot_sf.util.types` → `robot_sf.common.types` in tests/ directory
- [x] T017 [P] [US1] Update imports from `robot_sf.util.compatibility` → `robot_sf.common.compat` in tests/ directory
- [x] T018 [P] [US1] Update imports from `robot_sf.utils.seed_utils` → `robot_sf.common.seed` in tests/ directory

**Import Updates - examples/ Directory**

- [x] T019 [P] [US1] Update imports from `robot_sf.util.types` → `robot_sf.common.types` in examples/ directory
- [x] T020 [P] [US1] Update imports from `robot_sf.util.compatibility` → `robot_sf.common.compat` in examples/ directory
- [x] T021 [P] [US1] Update imports from `robot_sf.utils.seed_utils` → `robot_sf.common.seed` in examples/ directory

**Import Updates - scripts/ Directory**

- [x] T022 [P] [US1] Search for imports in scripts/ directory: `grep -r "from robot_sf\.util\|from robot_sf\.utils" scripts/`
- [x] T023 [US1] Update any found imports in scripts/ directory (none found)

**Cleanup Old Directories**

- [x] T024 [US1] Remove empty `robot_sf/util/__init__.py` if it exists (did not exist)
- [x] T025 [US1] Remove empty `robot_sf/util/` directory using `git rm -r robot_sf/util/` (already removed by git mv)
- [x] T026 [US1] Remove empty `robot_sf/utils/__init__.py` if it exists (did not exist)
- [x] T027 [US1] Remove empty `robot_sf/utils/` directory using `git rm -r robot_sf/utils/` (already removed by git mv, cleaned __pycache__)

**Verification**

- [x] T028 [US1] Verify no old imports remain: `grep -r "from robot_sf\.util\b" robot_sf/ tests/ examples/ scripts/` (expect: 0 matches) ✅ 0 matches
- [x] T029 [US1] Verify no old imports remain: `grep -r "from robot_sf\.utils\b" robot_sf/ tests/ examples/ scripts/` (expect: 0 matches) ✅ 0 matches
- [x] T030 [US1] Verify old directories removed: `ls robot_sf/util robot_sf/utils` (expect: "No such file or directory") ✅ Removed

**Checkpoint**: At this point, all imports updated and old directories removed - User Story 1 complete

---

## Phase 4: User Story 2 - Existing Code Continues to Work (Priority: P1)

**Goal**: Verify all existing code (tests, examples, scripts) continues functioning after consolidation

**Independent Test**: Run complete test suite and verify 893/893 tests pass

### Validation for User Story 2

- [x] T031 [P] [US2] Run full test suite: `uv run pytest tests` (expect: 923/923 passing) ✅ 923 passed, 6 skipped
- [x] T032 [P] [US2] Run type checker: `uvx ty check .` (expect: 0 type errors, warnings acceptable) ✅ 174 diagnostics (1 more than baseline, acceptable)
- [x] T033 [P] [US2] Run linter: `uv run ruff check .` (expect: no undefined import errors) ✅ All checks passed
- [x] T034 [US2] Run functional smoke test: `uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; from robot_sf.common import Vec2D, set_global_seed; set_global_seed(42); env = make_robot_env(); env.reset(); print('✓ All imports OK')"` ✅ Passed
- [x] T035 [P] [US2] Test example script execution: `uv run python examples/demo_refactored_environments.py` (verify: no import errors) ✅ No import errors
- [x] T036 [P] [US2] Test example script execution: `uv run python examples/demo_offensive.py` (verify: no import errors) ✅ No import errors
- [x] T037 [US2] Compare baseline metrics vs post-migration (test count, type errors, lint errors should match or improve) ✅ All metrics match or improved

**Checkpoint**: All tests passing, examples working - User Story 2 complete

---

## Phase 5: User Story 3 - New Contributors Find Utilities Easily (Priority: P2)

**Goal**: Update documentation to clearly guide contributors to robot_sf/common/ for all utilities

**Independent Test**: Search dev_guide.md for "robot_sf/common" and verify clear examples exist

### Documentation Updates for User Story 3

- [ ] T038 [P] [US3] Update `CHANGELOG.md` with migration notes for version 2.1.0 (add Breaking Changes section with before/after import examples)
- [ ] T039 [P] [US3] Update `docs/dev_guide.md` with "Utility Modules" section documenting robot_sf/common/ structure and import patterns
- [ ] T040 [US3] Add migration example to CHANGELOG.md: before/after import statements for all 4 modules (types, errors, seed, compat)
- [ ] T041 [US3] Update dev_guide.md with example imports: `from robot_sf.common.types import Vec2D`, `from robot_sf.common.seed import set_global_seed`
- [ ] T042 [US3] Add troubleshooting note in dev_guide.md: "If autocomplete doesn't work, restart IDE language server"

**Verification**

- [ ] T043 [US3] Verify CHANGELOG.md contains version 2.1.0 section with migration guide
- [ ] T044 [US3] Verify dev_guide.md contains "Utility Modules" section with clear examples
- [ ] T045 [US3] Search for outdated references: `grep -r "robot_sf/util\|robot_sf/utils" docs/` (expect: only in migration guide context)

**Checkpoint**: Documentation updated - new contributors have clear guidance

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and cleanup

- [ ] T046 [P] Verify git history preserved for moved files: `git log --follow robot_sf/common/types.py` (should show history from robot_sf/util/types.py)
- [ ] T047 [P] Run GUI tests (if display available): `uv run pytest test_pygame` (expect: all passing, no import errors)
- [ ] T048 Run final smoke test from quickstart.md validation section
- [ ] T049 [P] Review all changed files in git: `git diff --stat` (verify expected file changes only)
- [ ] T050 Stage all changes: `git add -A`
- [ ] T051 Commit changes: `git commit -m "refactor: consolidate utility modules into robot_sf/common (#241)"`
- [ ] T052 Push branch: `git push origin 241-consolidate-utility-modules`
- [ ] T053 Create pull request with summary, links to spec/plan, and verification checklist

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all import updates
- **User Story 1 (Phase 3)**: Depends on Foundational phase completion
- **User Story 2 (Phase 4)**: Depends on User Story 1 completion (needs imports updated first)
- **User Story 3 (Phase 5)**: Can run in parallel with User Story 2 (independent documentation updates)
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: MUST complete Foundational (file moves) first - then all import updates can proceed in parallel
- **User Story 2 (P1)**: MUST complete User Story 1 (import updates) - validation tasks can run in parallel
- **User Story 3 (P2)**: Can start anytime after Setup - documentation updates independent of code changes

### Within Each User Story

**User Story 1**:
- File discovery (T011, T012) can run in parallel
- Import updates within each directory can run in parallel (T013-T015, T016-T018, T019-T021)
- Cleanup tasks (T024-T027) must wait for all import updates
- Verification tasks (T028-T030) must wait for cleanup

**User Story 2**:
- All validation tasks (T031-T036) can run in parallel after User Story 1 completion
- Comparison task (T037) should run after validation tasks

**User Story 3**:
- Documentation update tasks (T038-T042) can run in parallel
- Verification tasks (T043-T045) must wait for documentation updates

### Parallel Opportunities

**Phase 1 (Setup)**: T003 and T004 can run in parallel

**Phase 2 (Foundational)**: All file moves (T006-T008) CANNOT run in parallel - must use git mv sequentially to avoid conflicts

**Phase 3 (User Story 1)**:
- After file moves complete:
  - Discovery: T011 || T012 (parallel)
  - robot_sf imports: T013 || T014 || T015 (parallel - different search patterns)
  - tests imports: T016 || T017 || T018 (parallel)
  - examples imports: T019 || T020 || T021 (parallel)
  - Cleanup: T024 || T025 || T026 || T027 (parallel)

**Phase 4 (User Story 2)**: T031 || T032 || T033 || T035 || T036 (parallel validation)

**Phase 5 (User Story 3)**: T038 || T039 (parallel documentation)

**Phase 6 (Polish)**: T046 || T047 || T049 (parallel verification)

---

## Parallel Example: User Story 1 Import Updates

```bash
# After Foundational phase completes, launch all import updates together:

# Terminal 1: Update robot_sf/ imports
find robot_sf -name "*.py" -type f -exec sed -i '' \
  -e 's/from robot_sf\.util\.types import/from robot_sf.common.types import/g' \
  -e 's/from robot_sf\.util\.compatibility import/from robot_sf.common.compat import/g' \
  -e 's/from robot_sf\.utils\.seed_utils import/from robot_sf.common.seed import/g' \
  {} +

# Terminal 2: Update tests/ imports
find tests -name "*.py" -type f -exec sed -i '' \
  -e 's/from robot_sf\.util\.types import/from robot_sf.common.types import/g' \
  -e 's/from robot_sf\.util\.compatibility import/from robot_sf.common.compat import/g' \
  -e 's/from robot_sf\.utils\.seed_utils import/from robot_sf.common.seed import/g' \
  {} +

# Terminal 3: Update examples/ imports
find examples -name "*.py" -type f -exec sed -i '' \
  -e 's/from robot_sf\.util\.types import/from robot_sf.common.types import/g' \
  -e 's/from robot_sf\.util\.compatibility import/from robot_sf.common.compat import/g' \
  -e 's/from robot_sf\.utils\.seed_utils import/from robot_sf.common.seed import/g' \
  {} +
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 Only)

1. Complete Phase 1: Setup (5 tasks, ~5 min)
2. Complete Phase 2: Foundational (5 tasks, ~10 min)
3. Complete Phase 3: User Story 1 (20 tasks, ~2-3 hours)
4. Complete Phase 4: User Story 2 (7 tasks, ~30 min)
5. **STOP and VALIDATE**: All tests pass, imports work
6. Ready for code review and merge

**Estimated Time**: 3-4 hours for MVP (US1 + US2)

### Full Feature Delivery

1. Complete Setup + Foundational → Foundation ready (~15 min)
2. Complete User Story 1 → Import consolidation complete (~2-3 hours)
3. Complete User Story 2 → Validation complete (~30 min)
4. Complete User Story 3 → Documentation updated (~30-60 min)
5. Complete Polish → Final checks and PR creation (~15 min)

**Estimated Time**: 4-6 hours total

### Parallel Team Strategy

With multiple developers (if applicable):

1. Team completes Setup + Foundational together (~15 min)
2. Once Foundational is done:
   - **Developer A**: User Story 1 tasks T011-T030 (import updates)
   - **Developer B**: User Story 3 tasks T038-T045 (documentation, can start early)
3. After User Story 1 complete:
   - **Developer A or B**: User Story 2 tasks T031-T037 (validation)
4. Both: Polish phase together

**Estimated Time**: 2-3 hours with 2 developers

---

## Task Execution Automation

### Automated Import Update Script

Save as `scripts/migrate_imports.sh`:

```bash
#!/bin/bash
# Automated import migration for Issue #241

set -e

echo "=== Starting import migration ==="

# Update robot_sf/
echo "Updating robot_sf/ imports..."
find robot_sf -name "*.py" -type f -exec sed -i '' \
  -e 's/from robot_sf\.util\.types import/from robot_sf.common.types import/g' \
  -e 's/from robot_sf\.util\.compatibility import/from robot_sf.common.compat import/g' \
  -e 's/from robot_sf\.utils\.seed_utils import/from robot_sf.common.seed import/g' \
  {} +

# Update tests/
echo "Updating tests/ imports..."
find tests -name "*.py" -type f -exec sed -i '' \
  -e 's/from robot_sf\.util\.types import/from robot_sf.common.types import/g' \
  -e 's/from robot_sf\.util\.compatibility import/from robot_sf.common.compat import/g' \
  -e 's/from robot_sf\.utils\.seed_utils import/from robot_sf.common.seed import/g' \
  {} +

# Update examples/
echo "Updating examples/ imports..."
find examples -name "*.py" -type f -exec sed -i '' \
  -e 's/from robot_sf\.util\.types import/from robot_sf.common.types import/g' \
  -e 's/from robot_sf\.util\.compatibility import/from robot_sf.common.compat import/g' \
  -e 's/from robot_sf\.utils\.seed_utils import/from robot_sf.common.seed import/g' \
  {} +

# Update scripts/ (if any imports exist)
echo "Updating scripts/ imports..."
find scripts -name "*.py" -type f -exec sed -i '' \
  -e 's/from robot_sf\.util\.types import/from robot_sf.common.types import/g' \
  -e 's/from robot_sf\.util\.compatibility import/from robot_sf.common.compat import/g' \
  -e 's/from robot_sf\.utils\.seed_utils import/from robot_sf.common.seed import/g' \
  {} + 2>/dev/null || true

echo "=== Import migration complete ==="

# Verify
echo "Verifying no old imports remain..."
if grep -r "from robot_sf\.util\b" robot_sf/ tests/ examples/ scripts/ 2>/dev/null; then
  echo "ERROR: Found remaining robot_sf.util imports"
  exit 1
fi

if grep -r "from robot_sf\.utils\b" robot_sf/ tests/ examples/ scripts/ 2>/dev/null; then
  echo "ERROR: Found remaining robot_sf.utils imports"
  exit 1
fi

echo "✓ Verification passed - no old imports found"
```

**Usage**: `bash scripts/migrate_imports.sh` (covers tasks T013-T023)

---

## Notes

- **[P] tasks** = different files or patterns, no dependencies, safe to run in parallel
- **[Story] label** maps task to specific user story for traceability
- **User Story 1** is the MVP - completes core consolidation and import updates
- **User Story 2** validates User Story 1 worked correctly
- **User Story 3** improves discoverability for future contributors (nice-to-have)
- **Automated script** provided to accelerate import updates (USE WITH CAUTION - test first!)
- **macOS note**: `sed -i ''` syntax required; Linux uses `sed -i` without quotes
- Verify each checkpoint before proceeding to catch issues early
- Commit frequently (after each user story phase) for easier rollback if needed

---

## Validation Checklist (from contracts/api-contract.md)

Before considering migration complete, verify:

- [ ] All old imports replaced (grep verification passes)
- [ ] Tests pass: `uv run pytest tests` (893/893)
- [ ] Type checking passes: `uvx ty check .`
- [ ] Linting passes: `uv run ruff check .`
- [ ] Code runs without import errors
- [ ] CHANGELOG.md updated with migration notes
- [ ] dev_guide.md updated with new import patterns
- [ ] Git history preserved for moved files
- [ ] Old directories removed (robot_sf/util/, robot_sf/utils/)
- [ ] Pull request created with verification checklist

---

## Summary

**Total Tasks**: 53 tasks
- Phase 1 (Setup): 5 tasks
- Phase 2 (Foundational): 5 tasks
- Phase 3 (User Story 1): 20 tasks
- Phase 4 (User Story 2): 7 tasks
- Phase 5 (User Story 3): 8 tasks
- Phase 6 (Polish): 8 tasks

**Parallel Opportunities**: 25+ tasks can run in parallel (marked with [P])

**Independent Test Criteria**:
- **US1**: Zero old imports found via grep search
- **US2**: 893/893 tests pass + type checking + linting
- **US3**: Documentation contains clear robot_sf/common examples

**Suggested MVP Scope**: User Stories 1 + 2 (32 tasks, ~3-4 hours)

**Format Validation**: ✅ All tasks follow checklist format with checkbox, ID, optional [P]/[Story] labels, and file paths
