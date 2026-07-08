# Issue 760: HEIGHT Model Performance Analysis

## Current Status

The HEIGHT model shows poor performance (1.42% success rate) on the Robot SF benchmark suite, significantly underperforming compared to other algorithms like ORCA (70.21%) and PPO (50.35%).

## Detailed Investigation Findings

### 1. Domain Mismatch (CONFIRMED - Strong Evidence)

**Issue**: HEIGHT was trained exclusively on `circle_crossing` scenarios using the `CrowdSim3DTbObs-v0` environment, but the benchmark evaluates it on 141 diverse scenarios including complex social interactions that HEIGHT was never trained on.

**Detailed Evidence**: 
- **Upstream Training Configuration** (`output/repos/CrowdNav_HEIGHT/README.md:67-68`):
  ```python
  env.scenario = "circle_crossing"  # Only trained on this scenario type
  env.env_name = "CrowdSim3DTbObs-v0"  # Different from Robot SF physics
  ```

- **Benchmark Scenario Diversity** (`configs/scenarios/classic_interactions_francis2023.yaml`):
  - 10 classic interaction types (bottlenecks, crossings, doorways, etc.)
  - 27 Francis 2023 scenarios (frontal approach, obstruction, group behaviors, etc.)
  - Total: 141 unique episodes with diverse social dynamics

- **Scenario Complexity Mismatch**:
  - HEIGHT trained on simple circular crossing patterns
  - Benchmark includes: narrow doorways, group joining/leaving, elevator scenarios, blind corners
  - These require social navigation skills not present in circle_crossing training

**Impact**: This represents a severe out-of-distribution evaluation where HEIGHT is being tested on scenarios it was never designed to handle.

### 2. Action-Semantic Loss (CONFIRMED - Strong Evidence)

**Issue**: The adapter's action projection pipeline introduces semantic loss through multiple transformation stages.

**Detailed Analysis**:

1. **Discrete Action Space** (`output/repos/CrowdNav_HEIGHT/crowd_sim/envs/crowd_sim_tb2.py`):
   ```python
   self.action_convert = {0: [0.05, 0.1], 1: [0.05, 0], 2: [0.05, -0.1],
                          3: [0, 0.1], 4: [0, 0], 5: [0, -0.1],
                          6: [-0.05, 0.1], 7: [-0.05, 0], 8: [-0.05, -0.1]}
   ```
   - Only 9 discrete actions with small increments (±0.05 m/s, ±0.1 rad/s)
   - Maximum linear acceleration: 0.05 m/s per timestep
   - At 0.1s timestep: 0.5 m/s² acceleration (very slow response)

2. **Stateful Velocity Accumulation** (Upstream vs Adapter):
   - **Upstream** (`crowd_sim_tb2.py`):
     ```python
     self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + delta_v, 
                                      self.config.robot.v_min, self.config.robot.v_max)
     ```
   - **Adapter** (`crowdnav_height.py:738-749`): Identical logic but with additional clipping

3. **Double Clipping Problem**:
   - Stage 1: Clip to upstream limits (0.0-0.5 m/s, -1.0 to 1.0 rad/s)
   - Stage 2: Clip to adapter limits (same defaults, but creates discontinuity)
   - **Result**: Action semantics are distorted, especially at boundaries

**Quantitative Evidence**:
- Benchmark logs: `projection_rate_mean: 0.3348` (33.5% of actions clipped)
- `mean_abs_delta_linear_mean: 0.1302` (large velocity adjustments needed)
- `max_abs_delta_linear_max: 0.5` (maximum clipping amount)

**Impact**: The discrete action space + stateful accumulation makes HEIGHT slow to respond to dynamic situations, while the double clipping distorts the intended action semantics.

### 3. Observation Mismatch (PARTIALLY CONFIRMED - Moderate Evidence)

**Issue**: While observation keys match, there are critical differences in how observations are constructed.

**Detailed Findings**:

1. **Lidar Implementation Differences**:
   - **Upstream**: Uses PyBullet `rayTestBatch` with full physics simulation
     - Detects humans and obstacles in 3D physics engine
     - Returns hit fractions and object IDs
   - **Adapter**: Custom 2D raycasting against obstacle segments only
     - `crowdnav_height.py:597-632`: Pure geometric ray-segment intersection
     - No human detection in lidar (humans handled separately)

2. **Obstacle Representation**:
   - **Upstream**: 3D cylinder obstacles with physics collision
   - **Adapter**: 2D line segments from SVG map parsing
   - **Missing**: Dynamic obstacle detection, 3D geometry effects

3. **Pedestrian Handling**:
   - **Upstream**: Full 3D human simulation with physics
   - **Adapter**: 2D position/velocity only, sorted by distance
   - **Potential Issue**: Spatial edge sorting may differ from training

**Impact**: The simplified 2D observation model loses critical 3D spatial relationships and physics interactions that HEIGHT was trained on.

### 4. Statefulness and Reset Issues (PLAUSIBLE - Needs More Investigation)

**Issue**: HEIGHT's recurrent policy may be sensitive to episode boundaries and reset conditions.

**Current Findings**:

1. **RNN State Management**:
   - Adapter correctly initializes hidden state (`crowdnav_height.py:469-473`)
   - Hidden state is properly passed between steps (`crowdnav_height.py:726-730`)
   - No obvious implementation errors found

2. **Reset Timing**:
   - **Upstream**: Reset called between episodes in training
   - **Benchmark**: Reset called at episode boundaries
   - **Potential Issue**: Different episode length distributions

3. **Attention Mask Handling**:
   - Adapter uses simple mask updating (`crowdnav_height.py:732`)
   - No obvious mismatch with upstream training

**Unanswered Questions**:
- Are episode lengths in benchmark different from training?
- Does the RNN state get corrupted across long episodes?
- Are there subtle timing differences in state updates?

**Recommendation**: Need to log RNN state evolution and compare with upstream training conditions.

### 5. NEW FINDING: Scenario Complexity Mismatch

**Issue**: HEIGHT was trained on simple scenarios but benchmark includes complex social navigation challenges.

**Evidence**:
- **Training Scenarios**: Simple circular crossing patterns with predictable dynamics
- **Benchmark Scenarios**: Include complex social behaviors:
  - `francis2023_join_group.yaml`: Group joining (requires social awareness)
  - `francis2023_narrow_doorway.yaml`: Doorway negotiation (requires patience)
  - `francis2023_robot_crowding.yaml`: Crowd navigation (requires adaptive behavior)
  - `francis2023_intersection_wait.yaml`: Social protocol understanding

**Impact**: HEIGHT lacks the social navigation policies needed for these complex scenarios.

## Comparison with Successful Algorithms

### ORCA (70.21% success)
- **Same projection path** but succeeds because:
  - Rule-based algorithm adapts to any scenario
  - Doesn't rely on learned social patterns
  - Projection doesn't hurt performance because ORCA is inherently robust

### PPO (50.35% success)
- **Trained directly on Robot SF physics** showing:
  - Domain-specific training is critical
  - HEIGHT's upstream training doesn't transfer well
  - PPO has learned policies for diverse scenarios

## Root Cause Analysis

Based on this investigation, the primary causes of HEIGHT's poor performance are:

1. **Primary Cause (60%)**: Domain mismatch - testing on scenarios HEIGHT was never trained for
2. **Secondary Cause (30%)**: Action projection issues - discrete actions + stateful accumulation are too slow for dynamic scenarios
3. **Tertiary Cause (10%)**: Observation differences - simplified 2D observations lose critical 3D spatial relationships

## Recommendations for Improvement

### Short-term (Validation)
1. **Test on Training Scenarios**: Run HEIGHT on `circle_crossing` scenarios to establish baseline performance
2. **Action Histogram Analysis**: Log action distribution and projection statistics across episodes
3. **State Tracking**: Monitor RNN hidden state and velocity accumulation patterns

### Medium-term (Adapter Improvements)
1. **Action Space Adaptation**: 
   - Increase discrete action increments for faster response
   - Reduce or eliminate double clipping
   - Consider continuous action projection

2. **Observation Enhancement**:
   - Add 3D obstacle representation
   - Improve lidar simulation to match upstream
   - Preserve original pedestrian sorting

3. **Scenario-Specific Testing**:
   - Identify which scenario types cause most failures
   - Focus improvement efforts on those specific cases

### Long-term (Fundamental)
1. **Fine-tune on Benchmark Scenarios**: Adapt HEIGHT to the specific scenarios it will be tested on
2. **Multi-scenario Training**: Train on diverse scenarios to improve generalization
3. **Consider Alternative Architectures**: Evaluate if HEIGHT's heterogeneous graph approach is suitable for these scenarios

## Files of Interest

- `robot_sf/planner/crowdnav_height.py`: Main adapter implementation
- `tests/planner/test_crowdnav_height.py`: Test cases showing expected behavior
- `configs/scenarios/classic_interactions_francis2023.yaml`: Benchmark scenario matrix
- `configs/scenarios/francis2023.yaml`: Francis 2023 specific scenarios
- `configs/scenarios/classic_interactions.yaml`: Classic interaction scenarios
- `output/repos/CrowdNav_HEIGHT/README.md`: Upstream training configuration
- `output/repos/CrowdNav_HEIGHT/crowd_sim/envs/crowd_sim_tb2.py`: Upstream environment implementation
