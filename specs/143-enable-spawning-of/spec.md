# Feature Specification: Single Pedestrian Spawning and Control

**Feature Branch**: `143-enable-spawning-of`  
**Created**: October 17, 2025  
**Status**: Draft  
**Input**: User description: "Enable spawning of single pedestrians with one goal or one trajectory in robot-sf"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Spawn Single Pedestrian with Fixed Goal (Priority: P1)

A researcher needs to create a reproducible test scenario where a single pedestrian walks from a specific starting position (e.g., [2.5, 1.2]) to a fixed goal position (e.g., [8.0, 5.0]) using the Social Force Model for navigation.

**Why this priority**: This is the foundational capability that enables controlled experiments and debugging. Without this, users cannot create deterministic single-pedestrian scenarios, which blocks reproducibility requirements for research validation.

**Independent Test**: Can be fully tested by defining a pedestrian with start and goal coordinates in a map configuration, running the simulation, and verifying the pedestrian reaches the goal position. Delivers immediate value for controlled testing scenarios.

**Acceptance Scenarios**:

1. **Given** a map configuration with a single pedestrian defined with start position [2.5, 1.2] and goal position [8.0, 5.0], **When** the simulation is initialized and run, **Then** the pedestrian spawns at the start position and navigates toward the goal using Social Force dynamics.
2. **Given** a pedestrian approaching its goal position, **When** the pedestrian reaches within goal proximity threshold, **Then** the pedestrian completes its journey and stops at the goal.
3. **Given** a simulation with obstacles between start and goal, **When** the pedestrian navigates, **Then** the pedestrian avoids obstacles while moving toward the goal.

---

### User Story 2 - Load Single Pedestrians from SVG/JSON Configuration (Priority: P1)

A researcher needs to define single pedestrians in the existing SVG map format or accompanying JSON metadata so they can be loaded and used without changing code.

**Why this priority**: This is the interface layer that makes the feature usable. Without configuration-based definition, users would need to modify code for each scenario, defeating the purpose of controlled experiments.

**Independent Test**: Can be tested by creating an SVG/JSON file with single pedestrian definitions, loading it through the map parser, and verifying the MapDefinition contains the correct pedestrian data.

**Acceptance Scenarios**:

1. **Given** an SVG map file with metadata defining single pedestrians, **When** the map is loaded via the parser, **Then** the MapDefinition object contains the single_pedestrians list with correct start, goal, and trajectory data.
2. **Given** a JSON configuration file with pedestrian definitions, **When** loaded through the configuration system, **Then** the pedestrian data is correctly parsed and made available to the simulator.
3. **Given** an invalid pedestrian definition (e.g., missing required fields), **When** the map is loaded, **Then** the parser raises a clear error message indicating the problem.

---

### User Story 3 - Spawn Single Pedestrian with Predefined Trajectory (Priority: P2)

A researcher wants to create a scenario where a pedestrian follows an exact predefined path (trajectory) consisting of multiple waypoints (e.g., [[0, 0], [1, 1], [2, 2], [3, 3]]) without goal-seeking behavior.

**Why this priority**: Enables precise control over pedestrian movement patterns for evaluating robot responses to specific pedestrian behaviors. Essential for testing edge cases and specific interaction patterns.

**Independent Test**: Can be tested by defining a pedestrian with a trajectory list, running the simulation, and verifying the pedestrian follows the exact path. Works independently of goal-based navigation.

**Acceptance Scenarios**:

1. **Given** a map configuration with a single pedestrian defined with trajectory [[0, 0], [1, 1], [2, 2], [3, 3]], **When** the simulation runs, **Then** the pedestrian follows the trajectory points in sequence without applying goal-seeking forces.
2. **Given** a pedestrian following a trajectory, **When** the pedestrian reaches the final waypoint, **Then** the pedestrian stops at that position.
3. **Given** a trajectory that passes through obstacles, **When** the simulation runs, **Then** the system handles the collision scenario appropriately (either warning or trajectory adjustment based on configuration).

---

### User Story 4 - Visualize Single Pedestrian Elements (Priority: P2)

A researcher running a simulation needs to visually distinguish single pedestrians, their starting positions, goal positions, and trajectories in the pygame visualization to understand and debug the scenario.

**Why this priority**: Visual feedback is critical for debugging and demonstrating scenarios. Without visualization, users cannot easily verify their configurations are correct.

**Independent Test**: Can be tested by loading a map with defined single pedestrians and verifying that start points (blue circles), goals (red crosses), and trajectories (polylines) appear correctly in the pygame window.

**Acceptance Scenarios**:

1. **Given** a simulation with defined single pedestrians, **When** the pygame visualization renders, **Then** pedestrian start positions are shown as distinct visual markers (e.g., blue circles).
2. **Given** a simulation with pedestrians that have goals, **When** the visualization renders, **Then** goal positions are shown as distinct markers (e.g., red crosses).
3. **Given** a simulation with pedestrians that have trajectories, **When** the visualization renders, **Then** trajectories are shown as polylines connecting the waypoints.

---

### User Story 5 - Define Multiple Single Pedestrians (Priority: P3)

A researcher wants to create a scenario with up to 4 individually controlled pedestrians, each with their own start position, goal, or trajectory, to test complex multi-agent interactions.

**Why this priority**: Extends the basic capability to more realistic scenarios with multiple controlled agents. Lower priority because single pedestrian must work first.

**Independent Test**: Can be tested by defining 4 pedestrians with different configurations (mix of goal-based and trajectory-based), running the simulation, and verifying all behave as specified independently.

**Acceptance Scenarios**:

1. **Given** a map with 4 single pedestrians defined (2 with goals, 2 with trajectories), **When** the simulation runs, **Then** all 4 pedestrians spawn and behave according to their individual configurations.
2. **Given** multiple pedestrians in proximity, **When** they move, **Then** they apply mutual Social Force interactions (avoid collisions with each other).
3. **Given** multiple pedestrians with unique IDs, **When** the simulation initializes, **Then** each pedestrian can be individually identified and tracked.

---

### Edge Cases

- What happens when a pedestrian's goal position is inside an obstacle?
- How does the system handle a trajectory that passes through impassable terrain?
- What if a pedestrian's start position overlaps with another pedestrian or robot?
- How does the system behave if trajectory waypoints are spaced very far apart or very close together?
- What happens when no goal or trajectory is specified (both are None)?
- How does the system handle duplicate pedestrian IDs?
- What if the trajectory list is empty?
- How does visualization handle scenarios with more pedestrians than can fit clearly on screen?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support defining individual pedestrians with unique identifiers (ID)
- **FR-002**: System MUST allow pedestrians to be spawned at explicit start positions defined as [x, y] coordinates
- **FR-003**: System MUST support pedestrians with a single goal position that they navigate toward using Social Force Model
- **FR-004**: System MUST support pedestrians with predefined trajectories (list of waypoint coordinates)
- **FR-005**: System MUST ensure pedestrians with trajectories do not apply goal-seeking forces
- **FR-006**: System MUST parse single pedestrian definitions from SVG map metadata or accompanying JSON configuration
- **FR-007**: System MUST extend the MapDefinition structure to include a list of single pedestrian definitions
- **FR-008**: System MUST integrate single pedestrians into the existing PySocialForce simulation alongside zone-spawned pedestrians
- **FR-009**: System MUST visualize pedestrian start positions, goals, and trajectories in the pygame rendering system
- **FR-010**: System MUST maintain compatibility with existing map structures and robot behavior (no breaking changes)
- **FR-011**: System MUST validate pedestrian definitions (e.g., ensure ID uniqueness, valid coordinates, mutually exclusive goal/trajectory)
- **FR-012**: System MUST provide clear error messages when pedestrian definitions are invalid or incomplete
- **FR-013**: System MUST support multiple single pedestrians in the same simulation (at least 4)
- **FR-014**: System MUST apply Social Force interactions between single pedestrians and other agents (robots, zone-spawned pedestrians)
- **FR-015**: System MUST ensure deterministic pedestrian behavior for a given random seed

### Key Entities *(include if feature involves data)*

- **SinglePedestrianDefinition**: Represents an individually controlled pedestrian with an ID, start position, and either a goal position or a trajectory (mutually exclusive). Contains validation logic to ensure consistency.
- **MapDefinition**: Extended to include a list of SinglePedestrianDefinition objects, maintaining relationships with existing map elements (obstacles, zones, routes).
- **PedestrianState**: Represents the runtime state of a pedestrian in the PySocialForce simulation, including position, velocity, and navigation target (goal or current trajectory waypoint).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can define and spawn a single pedestrian with a goal in under 2 minutes of configuration time (editing SVG/JSON)
- **SC-002**: Pedestrian navigation accuracy: pedestrians reach their goal position within the defined proximity threshold (typically < 0.5 meters) in 100% of valid scenarios
- **SC-003**: Visual clarity: 90% of users can correctly identify pedestrian starts, goals, and trajectories in the pygame visualization without additional documentation
- **SC-004**: Simulation performance: adding up to 4 single pedestrians does not increase per-step simulation time by more than 10% compared to baseline
- **SC-005**: Configuration validity: 100% of invalid pedestrian definitions (missing required fields, duplicate IDs, conflicting goal/trajectory) produce clear, actionable error messages
- **SC-006**: Reproducibility: running the same scenario with the same seed produces identical pedestrian behavior across 10 consecutive runs (bit-identical trajectories)
- **SC-007**: Backward compatibility: all existing maps and scenarios continue to function without modification after feature implementation
- **SC-008**: Documentation completeness: researchers can successfully create and run a single-pedestrian scenario following only the provided documentation and examples, verified by user testing with 3 independent researchers

## Assumptions

- The existing PySocialForce integration can accommodate individually spawned pedestrians alongside zone-based spawning
- SVG map format can be extended with metadata blocks without breaking existing parsers
- Pygame visualization system has sufficient rendering capacity for additional visual elements
- The Social Force Model is appropriate for goal-directed pedestrian navigation in the target use cases
- Users are familiar with the existing map configuration system and SVG/JSON editing
- The 40% performance penalty mentioned in the code comment for pedestrian obstacle forces applies to zone-spawned pedestrians and will be monitored for single pedestrians
- Trajectory-based pedestrians will use linear interpolation between waypoints unless otherwise specified
- The existing goal proximity threshold logic can be reused for single pedestrians

## Scope

### In Scope

- Parsing single pedestrian definitions from SVG metadata or JSON configuration
- Extending MapDefinition with SinglePedestrianDefinition structure
- Spawning individual pedestrians in the PySocialForce simulator
- Goal-based navigation using Social Force Model
- Trajectory-based movement following predefined waypoints
- Visualization of starts, goals, and trajectories in pygame
- Unit and integration tests for all new functionality
- Documentation updates and working examples (1-4 pedestrian scenarios)
- Validation and error handling for pedestrian definitions

### Out of Scope

- Advanced pedestrian AI behaviors beyond Social Force Model (e.g., adaptive re-planning, learning)
- Interactive GUI editor for placing pedestrians (remains manual SVG/JSON editing)
- Dynamic pedestrian spawning during simulation runtime (only at initialization)
- Pedestrian-specific collision detection beyond existing Social Force repulsion
- Custom force models for individual pedestrians
- Performance optimization of the underlying PySocialForce engine
- Multi-modal pedestrian behaviors (e.g., sitting, waiting, group formations for single pedestrians)
- Automatic trajectory generation or path planning for single pedestrians

## Dependencies

- PySocialForce library must support adding individual agents with explicit states
- Existing SVG map parser infrastructure (robot_sf/nav/svg_map_parser.py)
- Pygame rendering system (robot_sf/render/sim_view.py)
- MapDefinition and navigation module (robot_sf/nav/map_config.py)
- Simulator class and initialization logic (robot_sf/sim/simulator.py)

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| PySocialForce API doesn't support individual pedestrian insertion | High | Low | Review PySocialForce documentation early; implement adapter layer if needed |
| Performance degradation with multiple single pedestrians | Medium | Medium | Profile performance early; establish budgets per SC-004; optimize hotspots |
| SVG parsing complexity increases maintenance burden | Medium | Low | Keep metadata structure simple; provide JSON fallback; comprehensive tests |
| Visualization clutter with many pedestrians | Low | Medium | Implement optional visibility toggles; use distinct colors and sizes |
| Breaking changes to existing map format | High | Low | Maintain backward compatibility via optional fields; regression test suite |
| Users confused by goal vs. trajectory semantics | Medium | Medium | Clear documentation; validation errors; mutually exclusive enforcement |

## Open Questions

None at this time. All requirements have been specified with reasonable defaults based on existing architecture patterns in robot-sf.

## Related Work

- Existing pedestrian zone spawning system (robot_sf/ped_npc/ped_population.py)
- PySocialForce integration (robot_sf/sim/simulator.py, fast-pysf submodule)
- SVG map parsing and route definitions (robot_sf/nav/svg_map_parser.py)
- Pygame visualization of pedestrians and robots (robot_sf/render/sim_view.py)
- MapDefinition structure and navigation (robot_sf/nav/map_config.py)
