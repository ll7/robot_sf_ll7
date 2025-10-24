# Feature Specification: Metrics from Paper 2306.16740v4

**Feature Branch**: `144-implement-metrics-from`
**Created**: October 22, 2025
**Status**: Draft
**Input**: User description: "implement metrics from paper 2306.16740v4.pdf"

## Clarifications

### Session 2025-10-23

**Q1: Which specific metrics from paper 2306.16740v4 should be implemented?**

A: Metrics from Table 1 of the paper, including both NHT and SHT metrics:
- NHT: Success (S), Collision (C), Wall Collisions (WC), Agent Collisions (AC), Human Collisions (HC), Timeout (TO), Failure to progress (FP), Stalled time (ST), Time (T), Path length (PL), Success weighted by path length (SPL)
- SHT: Velocity (V_min/avg/max), Acceleration (A_min/avg/max), Jerk (J_min/avg/max), Clearing distance (CD_min/avg), Space compliance (SC), Minimum distance to human (DH_min), Time to collision (TTC), Aggregated Time (AT)

**Q2: What output schema format should be used for the metrics?**

A: Use the existing `robot_sf/benchmark/schemas/episode.schema.v1.json` format. Metrics should be added to the `metrics` object as key-value pairs where keys are metric names (lowercase with underscores, e.g., `success`, `collision_count`, `wall_collisions`, `velocity_avg`) and values are numeric (float or int). This maintains compatibility with existing benchmark infrastructure, aggregation tools, and CI validation. Follow the pattern established in `robot_sf/benchmark/metrics.py` where each metric is a pure function accepting `EpisodeData` and returning a numeric value.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compute Paper-Specific Metrics (Priority: P1)

Researchers need to evaluate robot social navigation performance using metrics defined in paper 2306.16740v4, enabling direct comparison with results published in that paper and other works that reference it.

**Why this priority**: Core functionality - without these metrics, users cannot validate their implementations against the paper's benchmarks or reproduce published results.

**Independent Test**: Can be fully tested by running a benchmark episode and verifying that all paper-defined metrics are computed and output in the expected format. Delivers immediate value by enabling paper-reproducible evaluations.

**Acceptance Scenarios**:

1. **Given** a completed robot navigation episode with trajectory and pedestrian data, **When** the metric computation is invoked, **Then** all metrics specified in paper 2306.16740v4 are calculated and returned
2. **Given** an episode with edge cases (no pedestrians, zero-length trajectory, boundary conditions), **When** metrics are computed, **Then** the system handles these gracefully without errors and returns appropriate values (e.g., NaN or 0)
3. **Given** multiple episodes with different scenarios, **When** metrics are aggregated, **Then** summary statistics (mean, median, confidence intervals) are correctly computed

---

### User Story 2 - Validate Against Paper Baselines (Priority: P2)

Researchers want to validate their metric implementations by comparing computed values against baseline scenarios or reference values from the paper, ensuring correctness before using metrics for evaluation.

**Why this priority**: Ensures metric implementation fidelity - users need confidence that metrics match paper definitions before trusting results.

**Independent Test**: Can be tested independently by creating synthetic test cases with known expected outcomes and verifying metric values match theoretical predictions or paper-reported values.

**Acceptance Scenarios**:

1. **Given** a synthetic test case with predetermined trajectory and pedestrian configurations, **When** metrics are computed, **Then** values match expected theoretical results within acceptable tolerance
2. **Given** reference scenarios from the paper (if available), **When** metrics are computed on these scenarios, **Then** values match or are within acceptable variance of paper-reported values
3. **Given** edge case scenarios (perfect navigation, worst-case collision), **When** metrics are computed, **Then** values correctly reflect boundary conditions

---

### User Story 3 - Export Metrics in Standard Format (Priority: P3)

Users need to export computed metrics in formats compatible with the paper's evaluation framework and common analysis tools, enabling integration with existing workflows and comparison across studies.

**Why this priority**: Enables interoperability - users can incorporate these metrics into existing analysis pipelines and compare with other work.

**Independent Test**: Can be tested by computing metrics and verifying output format matches schema specifications and can be successfully imported into analysis tools.

**Acceptance Scenarios**:

1. **Given** computed metrics for one or more episodes, **When** export is requested, **Then** metrics are output in JSON format matching `episode.schema.v1.json` with metric names as keys (e.g., `success`, `collision_count`, `velocity_avg`) and numeric values
2. **Given** batch metric computation across multiple episodes, **When** aggregated export is requested, **Then** summary statistics are included alongside individual episode metrics
3. **Given** metrics with missing or invalid values, **When** export is performed, **Then** these are clearly marked (NaN or null) and don't break the output format

---

### Edge Cases

- What happens when an episode has zero pedestrians (no social interaction)?
- How does the system handle trajectories with single timestep (no movement)?
- What happens when robot never moves or teleports (discontinuous trajectory)?
- How are metrics computed when pedestrian data is partially missing?
- What happens when force computations result in infinite or NaN values?
- How are metrics normalized when episode lengths vary dramatically?
- What happens when robot starts at goal (zero-length optimal path)?

## Metric Definitions from Paper 2306.16740v4

### NHT (Navigation/Hard Task) Metrics

These metrics evaluate basic navigation performance and task completion:

- **Success (S)**: Binary variable (boolean, [0,1]) - whether robot reached goal (averaged as Success Rate/SR)
- **Collision (C)**: Number of collisions (collision count, [0,∞)) - total collisions in trajectory (averaged as Collision Rate/CR)
- **Wall Collisions (WC)**: Number of collisions against walls (collision count, [0,∞))
- **Agent Collisions (AC)**: Number of collisions against other agents/robots (collision count, [0,∞))
- **Human Collisions (HC)**: Number of collisions against humans (collision count, [0,∞)), also called H-collisions (HB)
- **Timeout Before reaching goal (TO)**: Binary variable (timeout, [0,1]) - failure caused by timeout (has Time threshold parameter)
- **Failure to progress (FP)**: Number of failures (failure count, [0,∞)) - no progress toward goal for given time (has Distance & time thresholds parameters)
- **Stalled time (ST)**: Time duration (seconds, [0,∞)) - when robot speed magnitude falls below threshold (has Distance & time thresholds parameters)
- **Time to reach goal (T)**: Time duration (seconds, [0,∞)) - between task assignment and completion
- **Path length (PL)**: Trajectory length (meters, [0,∞))
- **Success weighted by path length (SPL)**: Success metric (unitless, [0,1]) - success weighted using normalized inverse path length

### SHT (Social/Human-aware Task) Metrics

These metrics evaluate quality and social aspects of navigation:

- **Velocity-based features (V_min, V_avg, V_max)**: Linear velocity statistics (m/s, [-∞,∞)) - minimum, average, and maximum linear velocity on trajectory
- **Linear acceleration based features (A_min, A_avg, A_max)**: Linear acceleration statistics (m/s², [-∞,∞)) - minimum, average, and maximum linear acceleration on trajectory
- **Movement jerk (J_min, J_avg, J_max)**: Jerk statistics (m/s³, [-∞,∞)) - minimum, average, and maximum linear jerk (second-order derivative of linear speed)
- **Clearing distance (CD_min, CD_avg)**: Distance statistics (meters, [0,∞)) - minimum and average distance to obstacles in trajectory
- **Space compliance (SC)**: Ratio metric (unitless, [0,1]) - ratio of trajectory with minimum distance to humans < given threshold (default 0.5m, Personal Space Compliance/PSC [26])
- **Minimum distance to human (DH_min)**: Distance (meters, [0,∞)) - minimum distance to a human in trajectory
- **Minimum time to collision (TTC)**: Time duration (seconds, [0,∞)) - minimum time to collision with human agent at any point in time (where PSC intersects in linear trajectory)
- **Aggregated Time (AT)**: Time duration (seconds, [0,∞)) - time taken for subset of cooperative agents to meet their goals (has Cooperative agents' set parameter)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST compute metrics from Table 1 of paper 2306.16740v4, organized into two categories:
  - **NHT (Navigation/Hard Task) Metrics**: Success (S), Collision (C), Wall Collisions (WC), Agent Collisions (AC), Human Collisions (HC), Timeout Before reaching goal (TO), Failure to progress (FP), Stalled time (ST), Time to reach goal (T), Path length (PL), Success weighted by path length (SPL)
  - **SHT (Social/Human-aware Task) Metrics**: Velocity-based features (V_min, V_avg, V_max), Linear acceleration (A_min, A_avg, A_max), Movement jerk (J_min, J_avg, J_max), Clearing distance (CD_min, CD_avg), Space compliance (SC), Minimum distance to human (DH_min), Minimum time to collision (TTC), Aggregated Time (AT)
- **FR-002**: System MUST handle edge cases (zero pedestrians, single-timestep episodes, missing data) without crashing
- **FR-003**: System MUST return metrics with appropriate data types (float, int, boolean) and handle special values (NaN, infinity) according to paper definitions
- **FR-004**: System MUST provide aggregation functions to compute summary statistics (mean, median, percentiles) across multiple episodes
- **FR-005**: System MUST validate metric computation correctness through unit tests covering normal cases and edge cases
- **FR-006**: System MUST export metrics in JSON format compatible with existing robot_sf benchmark schema
- **FR-007**: System MUST document each metric with formula, units, and interpretation guidance
- **FR-008**: System MUST integrate with existing `robot_sf/benchmark/metrics.py` module without breaking current functionality
- **FR-009**: System MUST provide clear error messages when required data is missing or invalid
- **FR-010**: System MUST compute metrics efficiently (< 100ms per episode for typical scenarios)

### Key Entities

- **Episode**: Complete navigation attempt with robot trajectory, pedestrian states, and termination condition
- **Metric**: Single quantitative measure (name, value, unit, timestamp)
- **MetricSuite**: Collection of related metrics computed from same episode data
- **AggregatedMetrics**: Summary statistics (mean, std, CI) across multiple episodes
- **PaperMetricConfig**: Configuration specifying which paper metrics to compute and their parameters

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All metrics from paper 2306.16740v4 are implemented and pass validation tests with 100% test coverage
- **SC-002**: Metric computation completes in under 100ms per episode for scenarios with up to 50 pedestrians
- **SC-003**: Metrics can be computed on 1000+ episode batches and aggregated in under 30 seconds
- **SC-004**: Metric values match theoretical expectations or paper baselines within 5% tolerance for test cases
- **SC-005**: System handles all identified edge cases without errors and returns valid output
- **SC-006**: Documentation includes clear formulas, units, and interpretation for each metric
- **SC-007**: Integration with existing benchmark framework requires zero breaking changes to current API
- **SC-008**: Exported metric files validate successfully against JSON schema

## Dependencies & Assumptions

### Dependencies

- Existing `robot_sf/benchmark/metrics.py` module and metric types
- Episode data schema with trajectory and pedestrian state information
- NumPy for numerical computations
- JSON schema validation for output format
- Existing SNQI (Social Navigation Quality Index) framework

### Assumptions

- Paper 2306.16740v4 defines metrics for social navigation evaluation (formula specifics need clarification)
- Episode data includes robot pose, velocity, pedestrian positions/velocities at each timestep
- Metrics are computed post-episode (not real-time during simulation)
- Standard SI units are used (meters, seconds, radians)
- Force computations follow Social Force Model conventions already in codebase
- Metric definitions are technology-agnostic and don't require specific planner implementation

## Technical Constraints

- Must maintain compatibility with existing robot_sf benchmark schema (v1)
- Must not introduce new external dependencies beyond current project requirements
- Performance must not degrade existing metric computation (< 10% overhead)
- Must support both single-episode and batch computation modes
- Must integrate with current testing framework (pytest)

## Out of Scope

- Real-time metric computation during episode execution
- Visualization or plotting of metric distributions
- Statistical comparison tests between different policies (separate analysis phase)
- Calibration of metric weights for composite indices
- Integration with external benchmarking platforms
- Metric computation for non-social navigation scenarios
- Modification of episode generation or simulation logic

## Notes

This specification addresses the implementation of metrics from research paper 2306.16740v4. The exact metric definitions, formulas, and validation criteria require clarification from the paper itself. This spec provides the framework for implementation assuming standard social navigation metrics (collision rates, comfort measures, path efficiency, etc.) commonly found in this research domain.

The implementation should extend the existing comprehensive metric suite already present in robot_sf (success rate, collision metrics, force-based comfort measures, path smoothness, SNQI composite index) rather than duplicating functionality.

Success S is boolean, SR (success_rate) is aggregated.
