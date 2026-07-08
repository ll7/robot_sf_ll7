# Tasks: Accelerate Slow Benchmark Tests

Feature Dir: /Users/lennart/git/robot_sf_ll7/specs/124-title-accelerate-slow  
Branch: 124-title-accelerate-slow  
Spec: spec.md  
Plan: plan.md

Legend:
- [P] = Parallelizable (independent files / order not coupled)
- Dependencies noted where sequencing required
- TDD principle: introduce/adjust tests & fixtures before optimization code where feasible

## Policy Reference
Soft threshold: 20s  
Hard timeout: 60s  
Report top N: 10  
Relax env var: ROBOT_SF_PERF_RELAX  
Potential enforce env var (future): ROBOT_SF_PERF_ENFORCE

## Task List

### Setup & Baseline Capture
T001. Capture current slow test baseline (durations) by running pytest with --durations=25 and store summarized output in `progress/slow_tests_pre.json` (script or manual capture). Output includes test node id and wall time.  
T002. Add a helper script `scripts/collect_slow_tests.py` that parses `pytest --durations` output (stdin or file) into JSON (fields: test_identifier, duration_seconds, timestamp). [P]

### Shared Helpers & Fixtures
T003. Create `tests/perf_utils/__init__.py` (module marker). [P]
T004. Implement `tests/perf_utils/policy.py` defining a lightweight `PerformanceBudgetPolicy` dataclass (soft, hard, report_count, relax_env_var) with method classify(duration)->("none"|"soft"|"hard"). [P]
T005. Implement `tests/perf_utils/reporting.py` providing function `generate_slow_test_report(samples, policy)` returning list of top N records and guidance text suggestions. [Depends: T004]
T006. Implement `tests/perf_utils/guidance.py` with heuristics converting a SlowTestRecord into guidance (episode reduction, horizon reduction, bootstrap disable, scenario minimization). [P]
T007. Implement `tests/perf_utils/minimal_matrix.py` providing helper `write_minimal_matrix(tmp_path)` returning path to YAML and content structure used by both resume & reproducibility tests. [P]
T008. Add `tests/conftest.py` fixture `perf_policy()` returning default policy object (20/60/10). Append (do not overwrite existing content). [Depends: T004]
T009. Add `tests/conftest.py` fixture `slow_test_collector` that records (nodeid, duration) using pytest hooks (pytest_runtest_call or pytest_runtest_teardown + timing) and at sessionfinish prints structured report unless ROBOT_SF_PERF_RELAX=1. [Depends: T004 T005 T006]

### Test Refactors (Benchmark Focus)
T010. Refactor `tests/benchmark_full/test_integration_reproducibility.py` to import and use `write_minimal_matrix` instead of inline YAML duplication (keep deterministic 2-episode semantics). [Depends: T007]
T011. Refactor `tests/benchmark_full/test_integration_resume.py` to use the shared minimal matrix helper ensuring expected episode count & resume no-op; keep timing assertions. [Depends: T007]
T012. Identify any additional heavy test under `tests/benchmark_full/` (e.g., full_classic) and introduce minimized scenario/horizon logic (create list in docstring of file). [P]
T013. Add per-test `@pytest.mark.timeout(60)` decorators to any remaining benchmark tests lacking explicit timeout. [P]
T014. Add performance assertion (wall time <20s unless ROBOT_SF_PERF_RELAX) to each heavy benchmark test after optimization, using a small timing wrapper. [Depends: T004]

### Slow Test Report Validation
T015. Create a synthetic slow test `tests/perf_utils/test_slow_report_fixture.py` that sleeps (e.g., 0.3s) and asserts it appears in the generated report; mark with ROBOT_SF_PERF_RELAX env var guard to avoid flakiness. [Depends: T009]
T016. Add test for guidance function producing expected keyword suggestions when a test exceeds soft threshold (e.g., includes 'reduce horizon'). [Depends: T006]

### Documentation & Changelog
T017. Update `docs/dev_guide.md` adding subsection "Per-Test Performance Budget" describing policy, env vars, guidance table. [Depends: T004 T005]
T018. Add link to new subsection in `docs/README.md`. [Depends: T017]
T019. Add CHANGELOG entry under Unreleased: summarize performance budget introduction and test reporting feature. [Depends: T017]

### Policy & Enforcement Enhancements
T020. Implement optional env var `ROBOT_SF_PERF_ENFORCE=1` logic in collector fixture to convert soft breaches into failures (raise pytest.ExitCode). [Depends: T009]
T021. Add test verifying enforce mode failure behavior by spawning a deliberately slow test and checking exit code (if feasible; otherwise document manual test). [Depends: T020]

### Post-Optimization Validation
T022. Re-run baseline capture (repeat T001) and write `progress/slow_tests_post.json`. [Depends: T010 T011 T012 T013 T014]
T023. Create comparison script `scripts/compare_slow_tests.py` to show before/after delta in durations and counts above 20s. [Depends: T022]

### Quality Gates & Cleanup
T024. Run lint/format (Ruff) and fix any style issues introduced in helpers. [Depends: code tasks]
T025. Run full test suite ensuring all new fixtures interact cleanly (no collection errors). [Depends: T009 T010 T011]
T026. Add inline comments in refactored tests explaining minimization rationale (avoid future regressions). [Depends: T010 T011]
T027. Ensure no duplication of minimal matrix logic (single source in `minimal_matrix.py`). [Depends: T010 T011]
T028. Final review: confirm no test >20s locally (sample few heavy ones) & documentation references match environment variable names. [Depends: T022]

### Stretch / Optional (Nice-to-Have)
T029. (Optional) Provide a CLI hook `python -m robot_sf.tools.slow_tests_report` to parse last run JSON and reprint guidance. [P]
T030. (Optional) Export slow test report as Markdown artifact under `results/slow_tests_report.md`. [Depends: T005]

## Parallelization Guidance
- Parallel group A: T002 T003 T004 T006 T007 (independent modules)
- Parallel group B: T010 T011 T012 (after helpers exist)
- Parallel group C: T015 T016 (after collector + guidance ready)
- Documentation tasks (T017-T019) can proceed once core fixtures (T009) stable.

## Dependency Graph (Simplified)
Policy (T004) → Reporting (T005) → Collector (T009) → Validation tests (T015) → Enforcement (T020)
Minimal Matrix (T007) → Test refactors (T010 T011)
Guidance (T006) → Report (T005) & Guidance tests (T016)

## Exit Criteria Checklist
- All heavy benchmark tests <20s soft budget
- Slow test report prints top 10 with guidance
- Relax env var honored; enforce env var optional
- Documentation & changelog updated
- Before/after comparison recorded

