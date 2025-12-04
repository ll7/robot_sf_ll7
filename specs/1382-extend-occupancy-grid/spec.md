# Feature Specification: Extended Occupancy Grid with Multi-Channel Support

**Feature Branch**: `1382-extend-occupancy-grid`  
**Created**: 2025-12-04  
**Status**: Draft  
**Input**: Extend occupancy grid with configurable channels for static obstacles and pedestrians, increase test coverage to 100%, add visual pygame tests, integrate with gymnasium observation space, and support point-of-interest queries for spawn validation

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Configure and Generate Multi-Channel Occupancy Grids (Priority: P1)

A navigation planner developer needs to create local occupancy grids around a robot with configurable resolution, size, and frame of reference. They want occupancy information organized as separate channels (static obstacles, pedestrians, dynamic agents) so they can independently weight or filter information in their planning algorithms.

**Why this priority**: This is the core feature—without this, users cannot generate grids at all. Every other use case depends on this capability.

**Independent Test**: Can be tested by creating a grid with specific parameters, verifying grid dimensions, resolution, and channel structure match the configuration.

**Acceptance Scenarios**:

1. **Given** a robot at position (0, 0) in a world with static walls, **When** creating a 10m × 10m occupancy grid at 0.1m resolution (100 × 100 cells) in ego-rotated frame, **Then** the grid correctly marks cells containing obstacles as occupied and free cells as unoccupied.

2. **Given** a configuration with separate channels enabled (static obstacles, pedestrians), **When** generating a grid in a scene with walls and 3 pedestrians, **Then** the output contains separate channel layers where static obstacles and pedestrians are represented independently.

3. **Given** ego-frame and world-frame options, **When** the robot is rotated 45°, **Then** ego-frame grids align obstacles to the robot's orientation while world-frame grids use fixed world coordinates.

4. **Given** dynamic pedestrian positions, **When** generating a grid at consecutive timesteps, **Then** pedestrian occupancy channels update to reflect new positions while static obstacles remain constant.

---

### User Story 2 - Gymnasium Observation Space Integration (Priority: P1)

An environment developer wants to include occupancy grids as part of the gymnasium observation space so that neural network policies can learn from grid-based representations alongside other observations.

**Why this priority**: Grid-based observations are critical for image-like learning in RL, and this directly supports the project's learning objectives.

**Independent Test**: Can be tested by creating an environment with occupancy grid observation, resetting the environment, and verifying the observation matches expected grid structure and values.

**Acceptance Scenarios**:

1. **Given** an environment configured with occupancy grid observation, **When** resetting the environment, **Then** the returned observation includes a grid array with expected shape (channels, height, width) and dtype (float32).

2. **Given** variable occupancy grid configurations (different sizes/resolutions), **When** creating environments with each configuration, **Then** the observation space is correctly defined and observations match the configuration.

3. **Given** multi-channel occupancy grids, **When** an agent steps through an environment, **Then** each channel in the observation independently reflects its information (obstacles, pedestrians, etc.) and updates appropriately.

---

### User Story 3 - Query Point-of-Interest Status for Spawn Validation (Priority: P2)

A scenario designer needs to programmatically check whether specific points or areas in the environment are free of obstacles and pedestrians, enabling automated spawn point validation and safety checks.

**Why this priority**: This supports automated scenario validation and improves robustness of the testing framework, but is not required for basic grid functionality.

**Independent Test**: Can be tested by querying multiple points in known free/occupied regions and verifying results.

**Acceptance Scenarios**:

1. **Given** a grid with static obstacles and pedestrians, **When** querying a point clearly inside free space, **Then** the query returns "free" (or equivalent boolean/float).

2. **Given** a grid, **When** querying a point occupied by a wall or pedestrian, **Then** the query returns "occupied" (or equivalent).

3. **Given** a circular region of interest around a spawn candidate, **When** querying whether the entire region is free, **Then** the query returns "safe to spawn" only if all cells in the region are unoccupied.

4. **Given** arbitrary world coordinates, **When** querying a point outside the grid bounds, **Then** the query either returns a safe default or raises a clear error explaining the out-of-bounds condition.

---

### User Story 4 - Visualize Occupancy Grids in Pygame (Priority: P2)

A developer debugging navigation or scenario setup wants to see the occupancy grid overlaid on the pygame visualization, with cells color-coded by occupancy state, helping them verify correctness and troubleshoot placement issues.

**Why this priority**: Visualization is essential for debugging and understanding grid behavior, but the system is functional without it.

**Independent Test**: Can be tested by enabling grid visualization in pygame, running a simulation, and verifying grid rendering (no crashed exceptions, grid visible on screen).

**Acceptance Scenarios**:

1. **Given** a pygame visualization with grid overlay enabled, **When** the simulation runs, **Then** a grid is visible overlaid on the scene with cells color-coded (e.g., light yellow for obstacles, transparent for free).

2. **Given** multi-channel grids, **When** toggling channel visibility in the visualization, **Then** only the selected channel(s) are rendered, allowing independent inspection.

3. **Given** ego-frame and world-frame grids, **When** switching frame modes in the visualization, **Then** the grid rotates/translates appropriately and displays the selected frame.

---

### User Story 5 - Achieve 100% Test Coverage (Priority: P1)

Quality assurance requires comprehensive test coverage of the occupancy grid module to ensure reliability and catch regressions early.

**Why this priority**: 100% coverage is a stated project requirement and ensures the module is well-tested before integration into production environments.

**Independent Test**: Can be verified by running coverage tools on the occupancy module and confirming all lines, branches, and edge cases are covered.

**Acceptance Scenarios**:

1. **Given** the occupancy.py module, **When** running coverage analysis, **Then** 100% of lines are executed by tests.

2. **Given** all grid generation paths, **When** running tests, **Then** boundary conditions, empty grids, full grids, and mixed scenarios are all tested.

3. **Given** error paths (invalid parameters, out-of-bounds queries), **When** running tests, **Then** all error conditions are triggered and assertions verify correct behavior.

### Edge Cases

- **Empty grid**: What happens when the environment contains no obstacles or pedestrians?
- **Fully occupied grid**: What happens when every cell is occupied?
- **Boundary queries**: How are queries exactly on or outside the grid boundary handled?
- **Very high resolution**: What is the performance impact when resolution is increased to extreme levels (e.g., 0.01m)?
- **Very low resolution**: What happens when grid resolution is very coarse (e.g., 1m per cell)?
- **Pedestrian at boundary**: How are pedestrians positioned exactly at the edge of the grid handled?
- **Rotated frame at cardinal angles**: Are there precision issues at 0°, 90°, 180°, 270°?
- **Concurrent frame changes**: If frame selection changes mid-simulation, is the transition handled correctly?

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: System MUST support configurable occupancy grid size (width/height in meters).
- **FR-002**: System MUST support configurable grid resolution (meters per cell).
- **FR-003**: System MUST support two frame modes: ego-rotated (robot-relative) and world-aligned (global coordinates).
- **FR-004**: System MUST include a separate channel for static obstacle occupancy.
- **FR-005**: System MUST include a separate channel for pedestrian occupancy (optional, configurable).
- **FR-006**: System MUST integrate occupancy grids as gymnasium observation space layers.
- **FR-007**: System MUST provide a point-of-interest query function to check if a specific world coordinate is free or occupied.
- **FR-008**: System MUST provide an area-of-interest query function to check if a circular or rectangular region is safely free of obstacles.
- **FR-009**: System MUST render occupancy grids in the pygame visualization with configurable color schemes.
- **FR-010**: System MUST allow toggling visualization of individual channels in pygame.
- **FR-011**: System MUST handle ego-frame rendering with correct rotation/translation relative to robot position.
- **FR-012**: System MUST correctly handle edge cases: empty grids, fully occupied grids, boundary conditions.
- **FR-013**: System MUST raise clear errors for invalid parameters (negative size, zero resolution, out-of-bounds queries).

### Key Entities

- **OccupancyGrid**: The main grid data structure holding occupancy values per cell, with metadata (size, resolution, frame, timestamp).
- **GridChannel**: An individual channel (static obstacles, pedestrians) representing one layer of information.
- **GridConfig**: Configuration object specifying grid parameters (size_m, resolution_m, frame, enabled_channels).
- **POIQuery**: A query object specifying a world coordinate or region to check for occupancy.
- **POIResult**: The result of a point/area query (occupied/free status, occupancy value, certainty).

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: Test coverage of `robot_sf/nav/occupancy.py` reaches and maintains 100% (line, branch, and condition coverage).

- **SC-002**: Occupancy grid generation completes in under 5ms for a 10m × 10m grid at 0.1m resolution with up to 100 pedestrians (measured on reference hardware).

- **SC-003**: POI queries complete in under 1ms per query for grids up to 20m × 20m at 0.1m resolution.

- **SC-004**: Pygame visualization with grid overlay runs at 30+ FPS (no performance degradation from grid rendering).

- **SC-005**: All gymnasium integration tests pass, verifying observation shape, dtype, and values match configuration.

- **SC-006**: Visual tests in pygame confirm grid rendering is correct (obstacles shown with yellow tint, free cells transparent, channels independently visualizable).

- **SC-007**: Documentation in `docs/dev/occupancy/` explains how to:
  - Create and configure occupancy grids
  - Integrate grids into gymnasium environments
  - Query grids for spawn validation
  - Visualize grids in pygame
  - Extend the occupancy module with custom channels

- **SC-008**: All edge cases pass (empty grids, fully occupied, boundary conditions, extreme resolutions).

- **SC-009**: Project no longer exhibits O(N) performance degradation with large obstacle sets; grid-based queries replace naive geometric checks where applicable.

- **SC-010**: Developers can spawn agents safely using grid-based validation with > 95% success rate (validated across 100 spawn attempts in diverse scenarios).

## Assumptions

- Grid cells use binary occupancy (occupied = 1.0, free = 0.0) or probability-based (0.0–1.0).
- Pedestrian occupancy is circle-based (position + radius).
- Static obstacles are represented as line segments or polygon boundaries (as per current map format).
- Frame transforms use standard 2D rotation matrices with no z-axis considerations (planar motion assumption).
- Gymnasium integration uses standard Box observation space for grid arrays.
- Pygame visualization assumes 8-bit RGB color model for rendering efficiency.
- POI queries default to conservative mode (cell is occupied if any pedestrian/obstacle occupies it).

## Out of Scope

- 3D occupancy grids or height-based occupancy.
- Real-time dynamic replay (grids are computed fresh per timestep).
- Serialization/deserialization of grids to disk.
- Integration with external mapping/SLAM systems.
- Gazebo or other non-pygame simulators.

## Related Issues/PRs

- Implements requirements from occupancy grid specification document.
- Extends and replaces legacy occupancy code in `robot_sf/nav/occupancy.py`.
- Integrates with gymnasium observation spaces defined in `robot_sf/gym_env/`.

## Open Questions

None at specification stage; all clarifications captured in Assumptions and Requirements.
