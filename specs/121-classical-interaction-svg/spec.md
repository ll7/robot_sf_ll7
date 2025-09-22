# Feature Specification: Classical Interaction SVG Map Pack

**Feature Branch**: `121-classical-interaction-svg`  
**Created**: 2025-09-19  
**Status**: Draft  
**Input**: User description: "Classical interaction SVG map pack with scenario matrix for robot-pedestrian micro-navigation patterns (crossing, head-on, overtaking, bottleneck, doorway, merging, T-intersection, group crossing)"

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a benchmarking user, I want a curated set of canonical social navigation interaction scenarios so that I can evaluate and compare robot navigation policies across well-understood micro-interaction patterns (crossing, head-on passing, overtaking, merging, queuing) without designing custom maps.

### Acceptance Scenarios
1. **Given** the repository installation, **When** I list available scenario matrices, **Then** I can find a `classic_interactions` scenario file documenting all archetypes.
2. **Given** the classic interactions matrix, **When** I run the benchmark runner with it, **Then** one JSONL episode line per (scenario variant × repeat × seed × algorithm) is generated with valid metrics.
3. **Given** the SVG map assets, **When** I open them, **Then** each shows clearly labeled obstacles and robot spawn/goal zones consistent with repository conventions.
4. **Given** a density variant (low/medium/high), **When** I inspect resulting episodes, **Then** pedestrian count/flows differ qualitatively (higher density → more interactions) while remaining valid (no persistent spawn collisions).
5. **Given** group-specific scenario (group_crossing), **When** I run episodes, **Then** pedestrians form groups (parameter present) and robot path exhibits negotiated avoidance.

### Edge Cases
- What happens if a map file is missing? → Scenario load MUST fail fast with a clear error referencing the missing filename.
- How is an invalid density label handled? → Loader MUST reject unknown density values (no silent fallback).
- What if spawn zones overlap obstacles? → MUST be avoided in provided assets; detection can be added later (non-blocking now) but maps shipped must avoid this condition.

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: The feature MUST provide ≥8 distinct SVG maps each modeling a unique interaction archetype: crossing, head_on_corridor, overtaking, bottleneck, doorway, merging, t_intersection, group_crossing.
- **FR-002**: Each map MUST include at least one `robot_spawn_zone` and one `robot_goal_zone` region clearly separated spatially.
- **FR-003**: Each map MUST use consistent obstacle labeling (`obstacle`) for all impassable geometry.
- **FR-004**: A consolidated scenario matrix file MUST enumerate all archetypes with density variants (low, medium, high) unless the archetype conceptually only supports a subset (documented inline comment if omitted).
- **FR-005**: Scenario entries MUST support configuration of flow pattern (e.g., uni, bi, converging) and group probability (only non-zero for group_crossing archetype unless extended later).
- **FR-006**: Running the benchmark with the matrix MUST produce schema-valid episode JSONL output without requiring code changes outside configuration and map assets.
- **FR-007**: Documentation MUST describe each archetype’s social interaction intent and expected differentiating metrics.
- **FR-008**: A smoke test MUST verify that all referenced map files exist and can be parsed (spawn zone + obstacle layers present).
- **FR-009**: The feature MUST NOT alter existing episode schema or baseline planner interfaces.
- **FR-010**: The matrix MUST allow deterministic seeding with existing seeding utilities (no new seed fields required).

### Key Entities *(data-focused)*
- **Interaction Archetype**: Conceptual scenario type characterized by spatial geometry + pedestrian flow pattern; attributes: name, map_filename, supported_density_levels, default_flow, optional_group_param.
- **Scenario Variant**: A concrete instantiation combining archetype + density + flow + group parameter + repeats.

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (HOW hidden; only assets + config WHAT)
- [x] Focused on user value (benchmark coverage & comparability)
- [x] Written for non-technical stakeholders (describes outcomes, not code)
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain (assumptions explicit)
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable (map count, scenario load, test existence)
- [x] Scope clearly bounded (assets + config only; no new metrics or planners)
- [x] Dependencies and assumptions identified (reuse existing loader, seeding utilities)

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (none outstanding)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed (final review pending planning)

