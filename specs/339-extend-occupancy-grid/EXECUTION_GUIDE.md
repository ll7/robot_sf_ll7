# Execution Guide: Occupancy Grid Feature Tasks

**Feature**: 339-extend-occupancy-grid  
**Generated**: 2025-12-04  
**Tasks**: 118 total | Phases: 8 | User Stories: 5  

---

## Quick Start (5 Minutes)

### 1. Review the Generated Artifacts

```bash
cd /Users/lennart/git/robot_sf_ll7/specs/339-extend-occupancy-grid/

# View the complete feature specification
cat spec.md              # Functional requirements, user stories, success criteria

# View the comprehensive implementation plan
cat plan.md             # Technical context, Constitution gates, project structure

# View the detailed research and design decisions
cat research.md         # 10 research findings that resolve all unknowns
cat data-model.md       # 5 core entities with fields and invariants
cat quickstart.md       # 7 usage patterns with code examples

# View the API contracts
cat contracts/occupancy_api.md  # Function signatures, parameters, performance guarantees

# View the GENERATED TASKS (START HERE FOR EXECUTION)
cat tasks.md            # 118 tasks organized by phase and user story

# View the task report (strategy and timelines)
cat TASKS_REPORT.md     # Parallel opportunities, MVP scope, risk assessment
```

### 2. Understand the Task Organization

**8 Phases**:
1. **Phase 1**: Setup (5 tasks) — Initialize branch and verify documentation
2. **Phase 2**: Foundational (9 tasks) — Core utilities, coordinates, rasterization **[BLOCKING]**
3. **Phase 3**: US1 (19 tasks) — Grid generation (MVP) **[P1 Priority]**
4. **Phase 4**: US2 (16 tasks) — Gymnasium integration **[P1 Priority]**
5. **Phase 5**: US3 (20 tasks) — POI queries **[Can parallel with Phase 4]**
6. **Phase 6**: US4 (20 tasks) — Visualization **[Can parallel with Phase 5]**
7. **Phase 7**: US5 (19 tasks) — 100% test coverage **[Depends on all stories]**
8. **Phase 8**: Polish (14 tasks) — Documentation and merge

### 3. Choose Your Execution Path

**Option A: MVP (4–6 weeks)**
```
Phases 1 → 2 → 3 → 4 → Polish subset
Deliverable: Occupancy grids + Gymnasium observation (RL-ready)
Tasks: 48 (T001–T048)
```

**Option B: Full Feature (8–12 weeks)**
```
Phases 1 → 2 → (3+4+5+6 parallel) → 7 → 8
Deliverable: Grids + Gymnasium + Queries + Visualization + 100% coverage
Tasks: 118 (T001–T118)
```

**Option C: Fast Parallel (6–9 weeks with team)**
```
Phases 1 & 2 sequential
Then Phases 3–6 in parallel (different team members on different user stories)
Then Phases 7 & 8 sequential
Deliverable: Full feature with optimized timeline
Tasks: 118 with 47 [P] parallelizable tasks
```

---

## Week-by-Week Breakdown (Full Feature, Option B)

### Week 1: Setup & Foundations (Phases 1–2)
**Tasks**: T001–T014 (14 tasks)
**Team**: 1–2 people
**Deliverable**: Infrastructure ready; all user stories unblocked

```bash
# Day 1–2: Phase 1 (Setup)
- [ ] T001 Initialize feature branch
- [ ] T005 Verify existing occupancy.py structure

# Day 2–5: Phase 2 (Foundational) — PARALLEL where possible
- [ ] T006 [P] Grid utility functions
- [ ] T007 [P] Frame transforms
- [ ] T008 [P] Obstacle rasterization
- [ ] T009 [P] Pedestrian rasterization
- [ ] T002 [P] Data model classes
- [ ] T003 [P] Query classes
- [ ] T004 [P] Constants
- [ ] T010 GridConfig in unified_config.py
- [ ] T011 RobotSimulationConfig extension
- [ ] T012 [P] Test fixtures
- [ ] T013 [P] SVG map fixtures
- [ ] T014 Logging setup

# Day 5: Checkpoint
✅ Phase 2 complete; ready to split team across user stories
```

### Weeks 2–4: User Story 1 (Phase 3)
**Tasks**: T015–T032 (19 tasks)
**Team**: 1–2 people (can be parallel with US2)
**Deliverable**: Grid generation, multi-channel, ego/world frames

```bash
# Week 2: Tests first (TDD approach)
- [ ] T015 [P] Empty grid test
- [ ] T016 [P] Single obstacle test
- [ ] T017 [P] Multiple pedestrian test
- [ ] T018 [P] Ego-frame test
- [ ] T019 [P] World-frame test
- [ ] T020 [P] Multi-channel test
- [ ] T021 [P] Grid update test
- [ ] T022 Integration test: real map
- [ ] T023 Integration test: FastPysfWrapper

# Week 3: Implementation
- [ ] T024 create_occupancy_grid() function
- [ ] T025 [P] Grid indexing
- [ ] T026 [P] Ego-frame transform
- [ ] T027 [P] World-frame pass-through
- [ ] T028 [P] Obstacle channel population
- [ ] T029 [P] Pedestrian channel population
- [ ] T030 OccupancyGrid.update() method

# Week 4: Polish & validation
- [ ] T031 Configuration validation
- [ ] T032 Error handling
- [ ] T033 Structured logging

# Week 4 Checkpoint
✅ US1 complete; grid generation works; tests pass
```

### Week 3–4 (Parallel): User Story 2 (Phase 4)
**Tasks**: T034–T048 (16 tasks)
**Team**: 1 person (parallel with US1)
**Deliverable**: Gymnasium observation integration

```bash
# Week 3: Parallel with US1 Week 2
- [ ] T034 [P] Box observation space test
- [ ] T035 [P] Grid-to-observation test
- [ ] T036 [P] Multi-channel observation test
- [ ] T037 [P] Variable config adaptation test
- [ ] T038 Environment reset test
- [ ] T039 Environment step test
- [ ] T040 StableBaselines3 training test

# Week 4: Implementation (after US1 infrastructure available)
- [ ] T041 grid_to_observation() function
- [ ] T042 [P] GridConfig in RobotSimulationConfig
- [ ] T043 [P] Modify env reset
- [ ] T044 [P] Modify env step
- [ ] T045 Observation space definition
- [ ] T046 Factory integration
- [ ] T047 Configuration validation
- [ ] T048 Logging

# Week 4 Checkpoint
✅ US1 + US2 complete; RL agents can train on occupancy observations
```

### Weeks 5–6 (Parallel): User Stories 3 & 4 (Phases 5–6)
**Tasks**: T049–T099 (51 tasks)
**Team**: 2 people (one per story) or sequential
**Deliverable**: Query API + Visualization

```bash
# Person A: US3 (Queries) — Weeks 5–6
Phase 5 tests (T049–T058) then implementation (T059–T067)

# Person B: US4 (Visualization) — Weeks 5–6
Phase 6 tests (T068–T075) then implementation (T076–T085)

# PARALLEL opportunities
All [P] tests can run in parallel
All [P] implementations can run in parallel

# Week 6 Checkpoint
✅ US3 + US4 complete; queries work; visualization works
```

### Week 7: User Story 5 (Phase 7)
**Tasks**: T086–T104 (19 tasks)
**Team**: 1–2 people
**Deliverable**: 100% test coverage, edge cases

```bash
# Week 7: Edge cases & coverage
- [ ] T086 [P] Empty grid edge case
- [ ] T087 [P] Fully occupied edge case
- [ ] T088 [P] High resolution edge case
- [ ] T089 [P] Low resolution edge case
- [ ] T090 [P] Boundary pedestrian edge case
- [ ] T091 [P] Cardinal angle rotations
- [ ] T092 [P] Frame transition edge case
- [ ] T093 [P] Single-cell grid edge case
- [ ] T094 [P] Zero pedestrians edge case
- [ ] T095 [P] Many pedestrians edge case
- [ ] T096 [P] Invalid config error test
- [ ] T097 [P] Out-of-bounds error test
- [ ] T098 Coverage verification
- [ ] T099 HTML coverage report

# Instrumentation (can parallel with edge cases)
- [ ] T100 Comprehensive docstrings
- [ ] T101 Boundary assertions
- [ ] T102 Type hints
- [ ] T103 Complexity reduction
- [ ] T104 Structured logging

# Week 7 Checkpoint
✅ 100% coverage achieved; all edge cases handled
```

### Week 8: Polish & Finalization (Phase 8)
**Tasks**: T105–T118 (14 tasks)
**Team**: 1–2 people
**Deliverable**: Documentation, examples, ready to merge

```bash
# Documentation (can parallel)
- [ ] T105 [P] docs/dev/occupancy/ usage guide
- [ ] T106 [P] API reference
- [ ] T107 [P] Configuration guide
- [ ] T108 [P] Troubleshooting section

# Examples (can parallel)
- [ ] T109 [P] Advanced example script
- [ ] T110 [P] Quickstart example
- [ ] T111 RL training example

# Final validation (sequential)
- [ ] T112 Update docs/README.md
- [ ] T113 [P] Performance validation
- [ ] T114 [P] Example smoke tests
- [ ] T115 Backward compatibility check
- [ ] T116 Update CHANGELOG.md
- [ ] T117 [P] Quality gates (ruff, type, tests)
- [ ] T118 Merge to main

# Week 8 Checkpoint
✅ Feature complete; merged to main
```

---

## Daily Execution Checklist

### Before Starting Each Phase

```bash
# 1. Ensure all prerequisites are met
cd /Users/lennart/git/robot_sf_ll7
git status  # Branch is 339-extend-occupancy-grid
ls -la specs/339-extend-occupancy-grid/tasks.md  # File exists

# 2. Review the phase in tasks.md
grep -A 20 "^## Phase N:" specs/339-extend-occupancy-grid/tasks.md

# 3. Check dependencies (can Phase N start?)
# Phase 1: Always
# Phase 2: After Phase 1
# Phase 3+: After Phase 2
# Phase 7: After Phases 3–6
# Phase 8: After Phase 7
```

### During Execution of Each Task

```bash
# 1. Copy task ID and description
# Example: T024 Implement create_occupancy_grid() function in robot_sf/nav/occupancy.py

# 2. Open the relevant file
code robot_sf/nav/occupancy.py

# 3. Implement according to specification
# Reference: specs/339-extend-occupancy-grid/contracts/occupancy_api.md
# Reference: specs/339-extend-occupancy-grid/quickstart.md for usage examples

# 4. Run local quality gates
uv run ruff check --fix .
uv run ruff format .
uv run pytest tests/test_occupancy*.py  # Run relevant tests

# 5. Update task checkbox in tasks.md
# Change: - [ ] T024 ...
# To:     - [x] T024 ...
```

### After Completing Each Phase

```bash
# 1. Verify all tasks in phase are marked complete
grep "^\- \[x\]" specs/339-extend-occupancy-grid/tasks.md | wc -l

# 2. Run full test suite
uv run pytest tests

# 3. Run quality gates
uv run ruff check . && uv run pylint robot_sf --errors-only && uvx ty check . --exit-zero

# 4. Commit progress
git add -A
git commit -m "Phase N complete: [summary of deliverables]"
git push origin 339-extend-occupancy-grid
```

---

## Task Format Reference

All tasks follow this exact format:

```
- [ ] T001 Initialize feature branch and verify specs/339-extend-occupancy-grid/ documentation is accessible
       ↑     ↑    ↑                    ↑
       |     |    |                    └─ File path (exact)
       |     |    └─ Task description
       |     └─ Task ID (T001–T118)
       └─ Checkbox (always - [ ])

With optional flags:

- [ ] T002 [P] Create base data model classes in robot_sf/nav/occupancy.py
       ↑         ↑
       |         └─ [P] = Parallelizable (can run simultaneously with other [P] tasks)
       └─ Checkbox

- [ ] T024 [US1] Implement create_occupancy_grid() function in robot_sf/nav/occupancy.py
            ↑
            └─ [US1] = User Story 1 (US1–US5 for story tasks; absent for Setup/Polish)
```

---

## Key Files to Keep Open

### During Execution

```
Primary reference:
  /specs/339-extend-occupancy-grid/tasks.md (current task list)
  /specs/339-extend-occupancy-grid/contracts/occupancy_api.md (function signatures)

Code guidance:
  /specs/339-extend-occupancy-grid/quickstart.md (usage examples)
  /specs/339-extend-occupancy-grid/data-model.md (entity definitions)

Architecture:
  /docs/dev_guide.md (project standards)
  /docs/dev/occupancy/Update_or_extend_occupancy.md (being updated)

Implementation:
  /robot_sf/nav/occupancy.py (primary module)
  /robot_sf/gym_env/unified_config.py (configuration)
  /robot_sf/render/sim_view.py (visualization)
```

---

## Progress Tracking

### Simple Completion Tracking

```bash
# Count completed tasks
grep -c "^\- \[x\]" specs/339-extend-occupancy-grid/tasks.md

# Count remaining tasks
grep -c "^\- \[ \]" specs/339-extend-occupancy-grid/tasks.md

# View completion by phase
grep -B 2 "^- \[" specs/339-extend-occupancy-grid/tasks.md | grep "Phase" | head -1
# Then look at checkbox status under that phase
```

### Detailed Progress Report

```bash
# Generate progress report
echo "=== Occupancy Grid Feature Progress ===" && \
echo "Total tasks: 118" && \
echo "Completed: $(grep -c "^\- \[x\]" specs/339-extend-occupancy-grid/tasks.md)" && \
echo "Remaining: $(grep -c "^\- \[ \]" specs/339-extend-occupancy-grid/tasks.md)" && \
echo "Completion %: $(( ($(grep -c "^\- \[x\]" specs/339-extend-occupancy-grid/tasks.md) * 100) / 118 ))"
```

---

## Common Issues & Solutions

### Issue 1: "Task file paths don't match my environment"
**Solution**: All paths in tasks.md are relative to repo root. Verify:
```bash
pwd  # Should be /Users/lennart/git/robot_sf_ll7
ls robot_sf/nav/occupancy.py  # Should exist
```

### Issue 2: "Phase 2 tasks take longer than expected"
**Solution**: This is the blocking phase. Don't skip tasks. If stuck:
```bash
# Review the research document
cat specs/339-extend-occupancy-grid/research.md

# Check the design decisions
cat specs/339-extend-occupancy-grid/plan.md
```

### Issue 3: "Can I skip a user story?"
**Solution**: User Stories 1 & 2 are MVP (P1 priority); skip 3–5 if timeline is tight:
- **Skip US3**: Queries are optional; grid generation + gymnasium sufficient for MVP
- **Skip US4**: Visualization is nice-to-have; not required for RL training
- **Skip US5**: Use continuous integration (CI) for coverage; manual testing sufficient for MVP

### Issue 4: "How do I parallelize tasks?"
**Solution**: Look for `[P]` flag in task ID:
```bash
# All these can run in parallel (different files)
- [ ] T006 [P] Grid utility functions in robot_sf/nav/occupancy.py
- [ ] T007 [P] Frame transform functions in robot_sf/nav/occupancy.py
- [ ] T008 [P] Obstacle rasterization in robot_sf/nav/occupancy.py
- [ ] T009 [P] Pedestrian rasterization in robot_sf/nav/occupancy.py

# Assign each to a different team member
```

---

## Completion Criteria Per Phase

### Phase 1: ✅ Setup
- [ ] Feature branch created and verified
- [ ] All specs/ artifacts accessible
- [ ] Old occupancy.py structure understood

### Phase 2: ✅ Foundational
- [ ] All utility functions implemented and tested
- [ ] Coordinate transforms working (ego/world)
- [ ] Rasterization functions implemented (obstacles, pedestrians)
- [ ] GridConfig dataclass defined with validation
- [ ] Test fixtures created
- [ ] Logging configured

### Phase 3: ✅ US1 (Grid Generation)
- [ ] `create_occupancy_grid()` function implemented
- [ ] OccupancyGrid.update() method working
- [ ] All unit tests pass (T015–T021)
- [ ] Integration tests pass with real maps (T022–T023)
- [ ] Ego-frame and world-frame modes verified

### Phase 4: ✅ US2 (Gymnasium)
- [ ] `grid_to_observation()` function working
- [ ] Observation space defined in environment
- [ ] Environment reset includes grid initialization
- [ ] Environment step updates grid
- [ ] StableBaselines3 training smoke test passes
- [ ] All gymnasium tests pass (T034–T040)

### Phase 5: ✅ US3 (Queries)
- [ ] `query_occupancy()` function implemented
- [ ] Point queries working (free/occupied)
- [ ] Circular AOI queries working
- [ ] Rectangular AOI queries working
- [ ] Out-of-bounds handling correct
- [ ] All query tests pass (T049–T058)

### Phase 6: ✅ US4 (Visualization)
- [ ] `render_grid_pygame()` function working
- [ ] Grid renders in pygame without crashes
- [ ] Color scheme correct (yellow obstacles, red pedestrians)
- [ ] Channel toggling interactive
- [ ] Ego-frame rotation correct
- [ ] World-frame alignment correct
- [ ] 30+ FPS maintained
- [ ] All visual tests pass (T068–T074)

### Phase 7: ✅ US5 (Coverage)
- [ ] All edge case tests pass (T086–T097)
- [ ] Coverage report shows 100% line coverage
- [ ] All error paths tested
- [ ] Docstrings complete
- [ ] Type hints added
- [ ] Complexity reduced

### Phase 8: ✅ Polish
- [ ] `docs/dev/occupancy/` updated with usage guide
- [ ] API reference complete
- [ ] Examples created and working
- [ ] Performance validation passed (<5ms gen, <1ms queries, 30+ FPS)
- [ ] All quality gates pass (ruff, pylint, type check, pytest)
- [ ] CHANGELOG.md updated
- [ ] Feature branch merged to main

---

## Questions?

### For Task Details
→ Review the specific task in `tasks.md` and check:
- Related test tasks
- Design document references (research.md, data-model.md, contracts/)
- Example code in quickstart.md

### For Architecture
→ Review `plan.md` and `research.md`:
- Technical context and Constitution gates
- Design decisions with rationale
- Implementation approach

### For API Details
→ Review `contracts/occupancy_api.md`:
- Function signatures and guarantees
- Parameter types and return types
- Performance targets

### For Getting Started
→ Review `quickstart.md`:
- 7 usage patterns with working code
- Common patterns and troubleshooting

---

## Ready to Start?

```bash
# 1. Switch to feature branch
git checkout 339-extend-occupancy-grid

# 2. Start Phase 1
code specs/339-extend-occupancy-grid/tasks.md

# 3. Mark T001 as in-progress
# Change: - [ ] T001 ...
# To:     - [ ] T001 ... (but make the checkbox indicate in-progress visually)

# 4. Begin execution
# Follow the task description and check contracts/occupancy_api.md for details
```

**Estimated full feature delivery**: **8–12 weeks** (118 tasks)  
**MVP delivery** (US1 + US2): **4–6 weeks** (48 tasks)  
**With parallel team**: **6–9 weeks** (optimal with 4+ people)

---

## Success!

Upon completion of all phases (Phase 1–8), the occupancy grid feature will be:

✅ **Complete**: All user stories implemented  
✅ **Tested**: 100% coverage with comprehensive edge cases  
✅ **Integrated**: Gymnasium observation, pygame visualization  
✅ **Documented**: Usage guides, API reference, examples  
✅ **Performance**: <5ms generation, <1ms queries, 30+ FPS visualization  
✅ **Merged**: Ready for production use  

---

**Feature Status**: ✅ Ready to Execute

All planning is complete. Grab a team member and start Phase 1 today! 🚀
