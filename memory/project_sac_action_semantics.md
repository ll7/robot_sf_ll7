---
name: project-sac-action-semantics
description: SAC planner action semantics mismatch — delta vs absolute velocity between training and benchmark; current implementation records semantics in SACPlannerConfig
metadata:
  type: project
  category: known-issues
  created: 2026-06-19
  issue: 790
  status: resolved
---

# Issue #790 SAC Planner Action Semantics Mismatch

**Status**: Resolved
**Issue**: [#790](https://github.com/ll7/robot_sf_ll7/issues/790)
**Branch**: (feature branch if tracked)
**Last Updated**: 2026-06-19

## Problem Statement

SAC (Soft Actor-Critic) planner implementation had an action semantics mismatch between:
- **Training environment**: Actions interpreted as **absolute** target velocities
- **Benchmark environment**: Actions interpreted as **delta** (relative change) velocities

**Symptom**: SAC model trained to convergence fails catastrophically in benchmark runs (high collision
rate, trajectory instability, low SNQI).

**Root cause**: Policy learned to output "set velocity to X", but planner interpreted actions as "change
velocity by X". Under benchmark contract, learned behavior was invalid.

---

## Resolution

### Current Implementation

`robot_sf/baselines/sac.py` declares `SACPlannerConfig.action_semantics` with explicit
`"delta"` and `"absolute"` modes. `SACPlanner._action_vec_to_dict()` preserves negative delta
actions in `"delta"` mode, while `"absolute"` mode clamps unicycle commands to `v_max` and
`omega_max`.

There is no standalone velocity-target action wrapper in the current tree. Future wrapper work
should either add the wrapper with tests, or keep this memory note pointed at the actual
`SACPlannerConfig` and `SACPlanner._action_vec_to_dict()` implementation.

**Impact**:
- Action semantics are explicitly declared in config rather than inferred.
- Unit tests cover the absolute-mode clamp behavior and related SAC planner compatibility paths.
- This note does not claim a benchmark result or trained-model recovery beyond the in-tree tests.

---

## Validation

### Test Coverage

- Unit tests: `tests/baselines/test_sac_planner.py`
- Implementation reference: `robot_sf/baselines/sac.py`
- Current proof boundary: implementation/test coverage for planner action conversion semantics.

### Benchmark Evidence

- **Grade**: test-backed implementation evidence only.
- **Benchmark result**: not established by this memory note.
- **Dedicated benchmark config**: not present in the current tree.

---

## Lessons Learned

### Hard Rule

**Action semantics must be explicit and validated at environment factory boundary.**

- **Why**: Implicit conversions hide training/benchmark contract violations; models fail silently
- **How to apply**: All planner implementations must declare `action_semantics` config; factory
  validates at instantiation time

### Prevention

1. Add integration test: train or load a model with one semantics, run under another, and either
   fail closed or convert explicitly.
2. Document action semantics in planner protocol docstring
3. Review code: Search for velocity/action calculations; check for implicit delta ↔ absolute transitions

---

## Related Issues

- Issue #790: SAC planner action semantics mismatch.

---

## Reference

- **Implementation**: `robot_sf/baselines/sac.py`
- **Test file**: `tests/baselines/test_sac_planner.py`
- **Config examples**: `configs/baselines/sac_gate_socnav_struct*.yaml`
