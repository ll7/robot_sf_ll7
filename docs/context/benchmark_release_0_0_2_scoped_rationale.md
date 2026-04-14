# Benchmark Release v0.0.2 Scoped (7-Planner Release) - Rationale

**Date**: 2026-04-14  
**Release ID**: `paper_experiment_matrix_7planners_v1_v0_0_2`  
**Release Tag**: `0.0.2-scoped`  
**Maturity**: pre-1.0 alpha  

## Executive Summary

Release v0.0.2 is published as a **scoped 7-planner release** (excluding socnav_bench) due to missing external dependencies. This is justified as a pragmatic decision for an alpha-stage library, prioritizing timely release of reproducible, stable planner benchmarks over blocked full coverage.

## Background

The full benchmark campaign targets 8 planners:
1. **goal** (core baseline) ✅ Reproducible
2. **social_force** (core baseline) ✅ Reproducible
3. **orca** (core baseline) ✅ Reproducible
4. **ppo** (experimental) ✅ Reproducible
5. **prediction_planner** (experimental) ✅ Reproducible
6. **socnav_sampling** (experimental) ✅ Reproducible
7. **sacadrl** (experimental) ✅ Reproducible
8. **socnav_bench** (experimental) ❌ **Excluded** - blocked by external assets

## Exclusion Rationale: socnav_bench

### Why socnav_bench Cannot Run

**socnav_bench** requires the **SocNavBench** external dataset, specifically:
- **skfmm** Python package (scipy-compatible fast marching method)
- **SD3DIS dataset** with traversibles pre-computed
- Data directory structure: `$SOCNAV_BENCHMARK_DIRS/*/traversibles/`

### Problem Statement

During preflight validation, socnav_bench fails with:
```
SocNavBench control pipeline parameters failed to load:
  [errno 2] No such file or directory: '<data_root>/traversibles'
```

This is **not a code regression** — it is a **missing external dataset dependency** that is:
- Not distributed with the repository
- Not installable via pip/conda
- Requires manual user download and setup via `docs/socnav_assets_setup.md`

### Historical Attempts (2026-04-13, 2026-04-14)

Scanned all campaign runs in the workspace for socnav_bench performance records:

| Attempt | Campaign ID | Status | Episodes | Reason |
|---------|------------|--------|----------|--------|
| 1 | paper_experiment_matrix_all_planners...20260413 | ❌ failed | 0 | `ModuleNotFoundError: No module named 'skfmm'` |
| 2 | paper_experiment_matrix_all_planners...20260414 | ❌ failed | 0 | Missing `traversibles` directories |

**Conclusion**: No successful socnav_bench runs exist in the workspace. The planner has **never** produced benchmark results (0 episodes in both attempts).

## Release Decision

Given the alpha status and external dependency blocker, we proceed with:

✅ **Decision**: Create scoped release excluding socnav_bench  
✅ **Justification**: Prioritize releasing 7 reproducible, stable planners over blocking on external assets  
✅ **Future Path**: Full 8-planner release contingent on asset staging (out-of-band process)  

## Scoped Release Configuration

**Campaign Config**: `configs/benchmarks/paper_experiment_matrix_7planners_v1.yaml`  
**Release Manifest**: `configs/benchmarks/releases/paper_experiment_matrix_7planners_v1_release_v0_0_2_scoped.yaml`  

Changes from all-planners config:
- Removed planner entry: `socnav_bench`
- Kept identical: scenario matrix, seed policy, SNQI weights, contract thresholds, kinematics
- Kept identical: workers, bootstrap, recording, resume policies
- **Result**: 7-planner benchmark with identical experimental design to full release

## Publication & Discoverability

### What Gets Published

✅ Campaign artifacts (7 successful planners)  
✅ SNQI contract validation report  
✅ Full markdown report with planner rankings and diagnostics  
✅ Video artifacts (if requested)  
✅ Publication bundle with release manifest  

### Release Notes

**Primary Note** (GitHub release):
```
## v0.0.2 (Scoped)

This release includes the **first stable benchmark of 7 core and experimental 
planners** for the Robot SF framework.

⚠️ **Note**: Release 0.0.2 is a scoped release (7 planners) due to missing 
external dependencies for socnav_bench. The full all-planners benchmark 
(including socnav_bench) requires manual setup of the SocNavBench SD3DIS dataset. 
See docs/socnav_assets_setup.md for setup instructions.

**Included Planners**: goal, social_force, orca, ppo, prediction_planner, 
socnav_sampling, sacadrl

**Benchmark Config**: paper_experiment_matrix_7planners_v1  
**Campaign Duration**: [runtime from campaign run]
```

**Disclosure in Docs**:
- Update `docs/benchmark.md` to note scoped release status
- Link to `socnav_assets_setup.md` in benchmark overview
- Document how to upgrade to full release once assets are staged

### Internal Documentation

- This file: rationale and history
- Campaign run logs: artifact paths in `output/benchmarks/camera_ready/`
- Manifest: `configs/benchmarks/releases/paper_experiment_matrix_7planners_v1_release_v0_0_2_scoped.yaml`

## Follow-Up Path: Full Release

To migrate to full 8-planner release in a future version:

1. **Stage Assets** (out-of-band):
   ```bash
   # User downloads SD3DIS dataset and runs:
   scripts/socnav/setup_socnav_assets.sh <data_root>
   ```

2. **Verify socnav_bench Planner**:
   ```bash
   uv run python scripts/validation/verify_planner.py --planner socnav_bench
   ```

3. **Create New Release Config**:
   - Revert to `paper_experiment_matrix_all_planners_v1.yaml`
   - Create manifest `paper_experiment_matrix_all_planners_v1_release_v0_0_3.yaml`
   - Run full campaign

4. **Publish as v0.0.3 (Full)**:
   - Release notes explain full coverage
   - Document v0.0.2 as predecessor (scoped)
   - Link both releases in GitHub with migration notes

## Risk Assessment

### Low-Risk Decision
- ✅ Code is unchanged; only configuration changes
- ✅ No regressions to 7-planner reproducibility
- ✅ SNQI normalization unaffected (7 planners sufficient for contract)
- ✅ Publication bundle integrity maintained
- ✅ Clear disclosure of scope in release notes

### Transparent Communication
- ✅ Release tag explicitly labeled `-scoped`
- ✅ Release ID distinct from all-planners release
- ✅ Manifest includes rationale comment
- ✅ Docs and GitHub notes disclose socnav_bench exclusion

### No Impact on Core Benchmark Value
- 7 planners represent diverse algorithm families (sampling, learning, force-based, graph search)
- All core baselines included (goal, social_force, orca)
- Experimental planners well-represented (ppo, prediction_planner, socnav_sampling, sacadrl)
- SNQI contract and rank alignment still enforce quality

## References

- **Scoped Campaign Config**: [`configs/benchmarks/paper_experiment_matrix_7planners_v1.yaml`](../paper_experiment_matrix_7planners_v1.yaml)
- **Scoped Release Manifest**: [`configs/benchmarks/releases/paper_experiment_matrix_7planners_v1_release_v0_0_2_scoped.yaml`](../releases/paper_experiment_matrix_7planners_v1_release_v0_0_2_scoped.yaml)
- **SocNav Asset Setup**: [`docs/socnav_assets_setup.md`](../socnav_assets_setup.md)
- **Benchmark Fallback Policy**: [`docs/context/issue_691_benchmark_fallback_policy.md`](issue_691_benchmark_fallback_policy.md)
- **Original All-Planners Config**: [`configs/benchmarks/paper_experiment_matrix_all_planners_v1.yaml`](../paper_experiment_matrix_all_planners_v1.yaml)
