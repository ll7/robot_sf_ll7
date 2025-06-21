# Refactoring: Extract Pedestrian Action Helper Function

## Overview

This document describes the extraction of a helper function to eliminate code duplication in pedestrian action preparation across multiple environment files.

## Problem Description

### Code Duplication Identified
The logic for preparing pedestrian action visualization data was duplicated across 4 environment files:

1. `robot_sf/gym_env/robot_env.py`
2. `robot_sf/gym_env/empty_robot_env.py`
3. `robot_sf/gym_env/pedestrian_env.py`
4. `robot_sf/gym_env/pedestrian_env_refactored.py`

### Duplicated Pattern
Each file contained identical code:

```python
# Prepare pedestrian action visualization
ped_actions = zip(
    self.simulator.pysf_sim.peds.pos(),
    self.simulator.pysf_sim.peds.pos() + self.simulator.pysf_sim.peds.vel(),
)
ped_actions_np = np.array([[pos, vel] for pos, vel in ped_actions])
```

This represented a clear violation of the DRY (Don't Repeat Yourself) principle and increased maintenance burden.

## Solution

### Helper Function Implementation
Created a new helper function in `robot_sf/gym_env/env_util.py`:

```python
def prepare_pedestrian_actions(simulator) -> np.ndarray:
    """
    Prepare pedestrian action visualization data.
    
    This helper function creates pedestrian action vectors for visualization
    by combining pedestrian positions with their velocity vectors.
    
    Args:
        simulator: The simulator object containing pysf_sim with pedestrian data
        
    Returns:
        np.ndarray: Array of shape (n_peds, 2, 2) where each pedestrian has
                   a start position [x, y] and end position [x, y] representing
                   their current position and position + velocity vector
    """
    ped_actions = zip(
        simulator.pysf_sim.peds.pos(),
        simulator.pysf_sim.peds.pos() + simulator.pysf_sim.peds.vel(),
    )
    return np.array([[pos, vel] for pos, vel in ped_actions])
```

### Refactoring Applied
Updated all 4 environment files to use the helper function:

**Before:**
```python
# Prepare pedestrian action visualization
ped_actions = zip(
    self.simulator.pysf_sim.peds.pos(),
    self.simulator.pysf_sim.peds.pos() + self.simulator.pysf_sim.peds.vel(),
)
ped_actions_np = np.array([[pos, vel] for pos, vel in ped_actions])
```

**After:**
```python
# Prepare pedestrian action visualization
ped_actions_np = prepare_pedestrian_actions(self.simulator)
```

## Benefits

### ✅ **Reduced Code Duplication**
- Eliminated 4 instances of identical code
- Centralized logic in a single, well-documented function
- Reduced maintenance burden

### ✅ **Improved Maintainability**
- Changes to pedestrian action logic only need to be made in one place
- Consistent behavior across all environments
- Clear documentation of the function's purpose and behavior

### ✅ **Better Code Quality**
- Follows DRY principle
- More readable code with descriptive function name
- Proper type hints and documentation

### ✅ **No Breaking Changes**
- Backward compatible - all environments work exactly as before
- Same return type and behavior
- All tests continue to pass

## Implementation Details

### Files Modified

**Helper Function Added:**
- `robot_sf/gym_env/env_util.py` - Added `prepare_pedestrian_actions()` function

**Updated Import Statements:**
- `robot_sf/gym_env/robot_env.py`
- `robot_sf/gym_env/empty_robot_env.py`
- `robot_sf/gym_env/pedestrian_env.py`
- `robot_sf/gym_env/pedestrian_env_refactored.py`

**Replaced Duplicated Code:**
- All 4 environment files now use the helper function

### Function Signature
```python
def prepare_pedestrian_actions(simulator) -> np.ndarray
```

**Parameters:**
- `simulator`: The simulator object containing `pysf_sim` with pedestrian data

**Returns:**
- `np.ndarray`: Shape `(n_peds, 2, 2)` containing start and end positions for each pedestrian

## Verification

### ✅ **Unit Testing**
- Helper function tested with mock data
- Verified correct output shape and values
- Confirmed identical behavior to original code

### ✅ **Integration Testing**
- All environments can be created and used successfully
- Environment reset and step operations work correctly
- No functional changes to environment behavior

### ✅ **Consistency Check**
- All 4 environments now use identical pedestrian action logic
- Consistent with velocity scaling feature implemented in visualization layer

## Future Considerations

### Additional Refactoring Opportunities
This refactoring demonstrates the value of identifying and eliminating code duplication. Similar patterns could be applied to other shared logic across environment files.

### Best Practices Reinforced
1. **DRY Principle**: Don't repeat yourself - extract common functionality
2. **Centralization**: Keep related logic in appropriate utility modules
3. **Documentation**: Clearly document function purpose and behavior
4. **Testing**: Verify refactoring doesn't change behavior

## Related Information

- **Related Fix**: [2x Speed VisualizableSimState Fix](../2x-speed-vissimstate-fix/README.md)
- **File**: `robot_sf/gym_env/env_util.py`
- **Function**: `prepare_pedestrian_actions()`
- **Lines of Code Reduced**: ~20 lines (5 lines × 4 files)
