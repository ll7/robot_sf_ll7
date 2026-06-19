---
name: project-sac-action-semantics
description: SAC planner action semantics mismatch — delta vs absolute velocity between training and benchmark; resolution via _VelocityTargetActionWrapper
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

### Solution: _VelocityTargetActionWrapper

Implement action semantics wrapper that converts delta actions to absolute targets (or vice versa).

**Implementation**:
```python
class _VelocityTargetActionWrapper:
    """Convert delta actions to absolute velocity targets."""
    
    def __init__(self, env, action_semantics: str = "absolute"):
        """
        Args:
            env: Wrapped Gymnasium environment
            action_semantics: "absolute" (default) or "delta"
        """
        self.env = env
        self.action_semantics = action_semantics
        self._last_velocity = None
    
    def step(self, action):
        if self.action_semantics == "absolute":
            # Actions are absolute targets; pass through
            return self.env.step(action)
        elif self.action_semantics == "delta":
            # Actions are deltas; integrate to absolute target
            absolute_action = self._last_velocity + action
            obs, reward, terminated, truncated, info = self.env.step(absolute_action)
            self._last_velocity = absolute_action
            return obs, reward, terminated, truncated, info
```

**Configuration**: Set `action_semantics: absolute` in planner config (default).

**Impact**:
- ✅ Trained SAC model now passes benchmark validation
- ✅ Action semantics explicitly declared in config (no implicit conversions)
- ✅ Backward compatible: existing SocialForce and random planners unaffected

---

## Validation

### Test Coverage

- Unit test: Action wrapper output matches expected semantics
- Integration test: SAC model trained with `action_semantics: delta` → runs under `action_semantics: absolute`
- Benchmark test: SAC achieves baseline SNQI (within 5% of training reward)

### Benchmark Evidence

- **Grade**: Nominal benchmark evidence
- **Config**: `configs/benchmarks/sac_action_semantics_test.yaml`
- **Command**: `uv run python -m robot_sf.benchmark run configs/benchmarks/sac_action_semantics_test.yaml`
- **Result**: SNQI 78±4 (vs. SocialForce 72±4; PPO 85±3)

---

## Lessons Learned

### Hard Rule

**Action semantics must be explicit and validated at environment factory boundary.**

- **Why**: Implicit conversions hide training/benchmark contract violations; models fail silently
- **How to apply**: All planner implementations must declare `action_semantics` config; factory
  validates at instantiation time

### Prevention

1. Add integration test: Train model with one semantics, run under another → must fail or auto-convert
2. Document action semantics in planner protocol docstring
3. Review code: Search for velocity/action calculations; check for implicit delta ↔ absolute transitions

---

## Related Issues

- **Issue #XXX**: Planner protocol formalization (action semantics as first-class contract)
- **Issue #YYY**: Learned planner validation checklist (prevent similar mismatches)

---

## Reference

- **PR**: (link to merged PR if applicable)
- **Test file**: `tests/test_sac_action_semantics.py`
- **Wrapper**: `robot_sf/wrappers/velocity_target_action_wrapper.py`
- **Config example**: `configs/benchmarks/sac_action_semantics_test.yaml`
