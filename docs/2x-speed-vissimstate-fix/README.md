# Fix: 2x Speed Issue in VisualizableSimState

## Overview

This document describes the resolution of a data integrity issue where pedestrian velocities were artificially doubled in saved simulation states, affecting downstream data analysis.

## Problem Description

### Root Cause
The issue originated in the gym environment files where `VisualizableSimState` objects were created. Pedestrian velocity vectors were being multiplied by 2:

```python
# Problematic code
self.simulator.pysf_sim.peds.pos() + self.simulator.pysf_sim.peds.vel() * 2
```

### Impact
1. **Data Corruption**: Saved states contained 2x actual pedestrian speeds
2. **Analysis Issues**: Research and data analysis worked with incorrect velocity data
3. **Inconsistency**: Diverged from fast-pysf reference implementation
4. **Architecture Violation**: Mixed visualization preferences with data representation

### Downstream Compensation
Data analysis code contained compensating logic that divided velocities by 2 to correct for the artificial inflation:

```python
# Compensating code in plot_dataset.py
velocity = velocity / 2  # See pedestrian_env.py -> ped_actions = ...
```

## Solution

### Complete Fix Strategy
The solution involved a coordinated fix across both data generation and analysis:

1. **Remove artificial 2x multiplier** from source data generation
2. **Remove compensating division** from data analysis
3. **Ensure end-to-end consistency** throughout the pipeline

### Implementation

**Environment Files (Data Generation):**
```python
# Before
ped_actions = zip(
    self.simulator.pysf_sim.peds.pos(),
    self.simulator.pysf_sim.peds.pos() + self.simulator.pysf_sim.peds.vel() * 2,  # ❌ Artificial 2x
)

# After  
ped_actions = zip(
    self.simulator.pysf_sim.peds.pos(),
    self.simulator.pysf_sim.peds.pos() + self.simulator.pysf_sim.peds.vel(),      # ✅ Actual velocity
)
```

**Data Analysis Files (Processing):**
```python
# Before (compensating for bad source data)
velocity = velocity / 2  # ❌ Compensating division

# After (working with correct source data)
velocity = velocity      # ✅ No manipulation needed
```

### Files Modified

**Environment Files:**
- `robot_sf/gym_env/robot_env.py`
- `robot_sf/gym_env/pedestrian_env.py`  
- `robot_sf/gym_env/empty_robot_env.py`
- `robot_sf/gym_env/pedestrian_env_refactored.py`

**Data Analysis Files:**
- `robot_sf/data_analysis/plot_dataset.py`

## Results

### Data Integrity Restored
- ✅ **Accurate Velocities**: Saved states contain actual pedestrian speeds
- ✅ **Research Quality**: Data analysis reflects true simulation behavior  
- ✅ **Consistency**: Aligns with fast-pysf reference implementation
- ✅ **Architecture**: Proper separation of data and visualization concerns

### Verification
- All existing tests continue to pass (170 tests)
- Data analysis functions work correctly with accurate data
- End-to-end pipeline maintains consistency
- No breaking changes to public APIs

## Future Considerations

### Visualization Enhancement (Optional)
If enhanced velocity visualization is desired for better visual clarity, implement it in the render layer only:

```python
def _augment_ped_actions(self, ped_actions: np.ndarray):
    """Draw pedestrian actions with optional visual enhancement."""
    for p1, p2 in ped_actions:
        # Optional: Apply 2x multiplier for visualization only
        if self.enhanced_velocity_display:
            direction = np.array(p2) - np.array(p1)
            p2 = np.array(p1) + direction * 2
            
        pygame.draw.line(self.screen, PED_ACTION_COLOR, 
                        self._scale_tuple(p1), self._scale_tuple(p2), width=3)
```

### Best Practices Established
1. **Separation of Concerns**: Keep visualization preferences separate from data representation
2. **Data Integrity**: Ensure saved states accurately reflect simulation reality
3. **Consistency**: Maintain compatibility with reference implementations
4. **Documentation**: Document architectural decisions and their reasoning

## Related Information

- **GitHub Issue**: [robot_sf_ll7/robot_env.py#L214-L215](https://github.com/ll7/robot_sf_ll7/blob/7c0d62798e9c45df957b8a3bdce7bd380ab68ac5/robot_sf/gym_env/robot_env.py#L214-L215)
- **Affected Research**: Julius Miller's data analysis workflows
- **Reference Implementation**: fast-pysf pedestrian simulation library
- **Progress Report**: `progress/2x-speed-vissimstate-fix/progress-report.md`
