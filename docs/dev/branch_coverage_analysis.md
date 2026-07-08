# Branch-Coverage Threshold Analysis Report

**Date**: 2026-07-09
**Method**: CPU-only pytest with `--cov-branch` on `robot_sf` and `pysocialforce`.
**Report script**: `scripts/dev/branch_coverage_report.py`
**Coverage data**: `output/coverage/coverage.json` (worktree-local, not committed)

## Overall

| Metric | Value |
|---|---|
| Branch coverage | 7.3% (2,418 / 33,248) |
| Line coverage | 24.5% (30,219 / 99,716) |
| robot_sf files tested | 654 |
| Files with 0% branch coverage | 462 (70.6%) |
| Test files skipped (missing optional deps) | 57 |

**Context caveat**: 57 test files could not collect (missing `torch`, `stable-baselines3`, `optuna`, `duckdb`, `pyarrow`). Modules exercised only by those tests report 0% branch coverage even though they may have tests in CI. This report reflects the CPU-only subset only.

## Per-Package Branch Coverage

| Package | Branch % | Covered | Total |
|---|---|---|---|
| robot_sf/sim | 63.5% | 216 | 340 |
| robot_sf/examples | 57.7% | 60 | 104 |
| robot_sf/common | 47.3% | 53 | 112 |
| robot_sf/sensor | 28.0% | 80 | 286 |
| robot_sf/robot | 27.2% | 25 | 92 |
| robot_sf/ped_npc | 26.6% | 82 | 308 |
| robot_sf/gym_env | 33.3% | 284 | 852 |
| robot_sf/research | 15.7% | 111 | 708 |
| robot_sf/nav | 14.7% | 213 | 1,450 |
| robot_sf/baselines | 10.5% | 62 | 592 |
| robot_sf/benchmark | 5.9% | 984 | 16,634 |
| robot_sf/training | 6.7% | 170 | 2,534 |
| robot_sf/prediction | 5.0% | 3 | 60 |
| robot_sf/render | 1.5% | 12 | 782 |
| robot_sf/planner | 0.9% | 36 | 3,938 |
| robot_sf/adversarial | 0.1% | 1 | 1,042 |
| robot_sf/analysis | 0.0% | 0 | 140 |
| robot_sf/analysis_workbench | 0.0% | 0 | 454 |
| robot_sf/scenario_certification | 0.0% | 0 | 744 |
| robot_sf/telemetry | 0.0% | 0 | 392 |
| robot_sf/maps | 0.0% | 0 | 556 |
| robot_sf/manual_control | 0.0% | 0 | 248 |
| robot_sf/feature_extractors | 0.0% | 0 | 118 |
| robot_sf/models | 0.0% | 0 | 124 |

## 10 Worst-Covered Evidence-Critical Modules

These modules are in packages that produce or validate benchmark, planner, or
scenario-certification evidence. All have 0% branch coverage in the CPU-only
run (57 test collection errors due to missing optional dependencies).

| # | Module | Branches | Lines |
|---|---|---|---|
| 1 | `robot_sf/planner/socnav.py` | 0/674 | 381/2,274 |
| 2 | `robot_sf/planner/hybrid_rule_local_planner.py` | 0/430 | 241/1,261 |
| 3 | `robot_sf/benchmark/map_runner.py` | 0/362 | 188/1,030 |
| 4 | `robot_sf/scenario_certification/perturbation_preflight.py` | 0/302 | 106/728 |
| 5 | `robot_sf/benchmark/map_runner_episode.py` | 0/300 | 112/797 |
| 6 | `robot_sf/analysis_workbench/trace_failure_predicates.py` | 0/294 | 113/667 |
| 7 | `robot_sf/benchmark/camera_ready/_config.py` | 0/244 | 37/431 |
| 8 | `robot_sf/benchmark/artifact_publication.py` | 0/228 | 96/557 |
| 9 | `robot_sf/benchmark/live_forecast_replay_gate.py` | 0/196 | 82/485 |
| 10 | `robot_sf/benchmark/snqi_scalarization_sensitivity.py` | 0/190 | 87/485 |

## Proposed Phased Threshold Schedule

### Phase 1 — Measure-only (current)

No threshold enforcement. Goal: establish a reproducible baseline and identify
blind spots. This report serves as the Phase 1 deliverable.

### Phase 2 — Floor at current baseline (~7%)

- Enforce: `--cov-fail-under=7` on the CPU test subset.
- Prevents regressions on already-tested code paths.
- **Target**: Q3 2026, after optional-dep tests are integrated into CPU CI.

### Phase 3 — Raise to 35%

- Focus on the 10 worst evidence-critical modules above.
- Priority: `benchmark/map_runner.py`, `planner/socnav.py`, `scenario_certification/perturbation_preflight.py`.
- **Target**: Q4 2026.

### Phase 4 — Raise to 50%

- Requires branch-coverage tests for planner, benchmark, and scenario_certification.
- Address the 462 robot_sf files with 0% branch coverage.
- **Target**: Q1 2027.

### Phase 5 — Target 70%+ (evidence-critical only)

- Enforce per-package thresholds on evidence-critical packages only.
- Non-evidence packages (render, manual_control, telemetry) excluded.
- **Target**: Q2 2027.

## Recommended Immediate Actions

1. Add `--cov-fail-under=7` to the CPU test runner to lock the baseline
   (matches the Phase 2 floor; must stay at or below the ~7.3% baseline or CI
   fails immediately — so 7, not 25).
2. Open follow-up issues for the 10 worst evidence-critical modules.
3. Integrate optional-dep tests into CPU CI with `--ignore` guards for
   GPU-only tests.

## Reproduction

```bash
# In any fresh worktree:
uv sync --all-extras
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests fast-pysf/tests \
    -m "not slow" --cov=robot_sf --cov=pysocialforce --cov-branch \
    --cov-report=json:output/coverage/coverage.json \
    --cov-report=term-missing -q --continue-on-collection-errors

# Generate the report:
uv run python scripts/dev/branch_coverage_report.py
```
