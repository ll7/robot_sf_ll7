# Feature Specification: Map Verification Workflow

**Feature Branch**: `001-map-verification`  
**Created**: 2025-11-20  
**Status**: Draft  
**Input**: User description: "Verify that all maps work as intended and surface actionable failures."

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

### User Story 1 - Single-command map audit (Priority: P1)

As a maps maintainer, I want a single command that validates every SVG map for structural, metadata, and spawn-point issues so that I can catch regressions locally before opening a PR.

**Why this priority**: Map asset issues currently surface only when examples/tests fail deep in the stack, wasting review time. A fast local audit prevents broken assets from entering the repo.

**Independent Test**: Run `uv run python scripts/validation/verify_maps.py --all` on a workstation with no other changes; the command completes with a PASS/FAIL summary and per-map diagnostics.

**Acceptance Scenarios**:

1. **Given** a clean checkout with existing maps, **When** the maintainer runs the audit command, **Then** the tool loads every map, validates geometry/metadata, and prints a concise summary with exit code 0 when all checks pass.
2. **Given** a map missing required spawn metadata, **When** the audit runs, **Then** the tool fails the map, emits the precise reason (missing `start_positions`), and returns exit code 1.

---

### User Story 2 - CI gate for map regressions (Priority: P2)

As a release engineer, I want CI to fail with actionable logs whenever a map fails verification so that bad assets never reach main.

**Why this priority**: Preventing regressions at merge time protects downstream consumers and keeps automated benchmarks stable.

**Independent Test**: Trigger the CI task (or VS Code "Check Code Quality" surrogate) on a branch with a purposely broken map; verify the job exits non-zero with attached diagnostics while succeeding when maps are valid.

**Acceptance Scenarios**:

1. **Given** CI runs on a PR, **When** all maps pass the verifier, **Then** the CI job completes within the performance budget and records a green status.
2. **Given** CI runs against a branch introducing an invalid polygon, **When** the verifier detects a self-intersection, **Then** CI fails with logs pointing to the file path, offending layer, and remediation hints.

---

### User Story 3 - Machine-readable verification manifest (Priority: P3)

As a tooling engineer, I want the verifier to emit a JSON manifest with per-map results so that downstream dashboards and docs can display verification health without scraping logs.

**Why this priority**: Structured output enables automation (docs badges, trend tracking) and supports future visualization efforts.

**Independent Test**: Run the verifier with `--output output/validation/map_verification.json`; inspect the file to confirm it contains map IDs, pass/fail state, rule identifiers, and timestamps.

**Acceptance Scenarios**:

1. **Given** the verifier finishes, **When** an output path is supplied, **Then** a JSON file is written with one record per map and schema-aligned fields.
2. **Given** a downstream script reads the manifest, **When** it filters by `status == "fail"`, **Then** it reliably lists the same maps flagged in console output without additional parsing.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

- What happens when a map references external assets (textures) that are missing locally? The verifier must flag the dependency and continue with other maps.
- How does the system handle gigantic SVGs (>5MB) that could overwhelm CI time limits? The verifier should time-box parsing, mark the map as needing optimization, and keep the run under the 60s hard limit.
- How are archived or experimental maps treated? The verifier should respect manifest flags (e.g., `ci_enabled: false`) to avoid failing on intentionally broken demos while still allowing opt-in checks.
- What if the environment factory cannot instantiate a map due to backend mismatch? The verifier should fall back to a dummy backend and record the failure reason explicitly.

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: The verifier MUST parse every SVG under `maps/svg_maps/` (excluding archived entries) and validate geometric consistency (closed polygons, non-intersecting walls, ordered layers).
- **FR-002**: The verifier MUST validate required metadata blocks (spawn points, goal regions, pedestrian zones) and fail fast with human-readable remediation steps when fields are missing or malformed.
- **FR-003**: The system MUST instantiate each map via `make_robot_env` (or appropriate factory) in headless mode to ensure runtime compatibility with the active simulator backend.
- **FR-004**: The CLI MUST provide configurable scope filters (by filename glob, manifest tag, or recently touched files) to support focused validation during incremental work.
- **FR-005**: The tool MUST emit structured JSON/JSONL output summarizing rule results, durations, and recommendations, respecting the artifact policy by defaulting to `output/validation/`.
- **FR-006**: The verifier MUST integrate with CI (task or script) and fail the job if any mandatory map check fails while still posting a summarized table in the logs.
- **FR-007**: The system MUST capture timing data per map and warn when individual validations exceed the defined soft performance budget (20s) to catch heavy assets.
- **FR-008**: The CLI MUST support `--fix` hooks for auto-remediable issues (e.g., sorting layer IDs) while leaving destructive actions behind an explicit confirmation flag.

*Clarifications still needed:*

- **FR-009**: Pedestrian-only maps MUST be instantiated via the dedicated pedestrian factory when their manifest/tag indicates that modality; all other maps continue to use the robot factory to ensure backend parity.
- **FR-010**: V1 scope is limited to local repository maps; remote bundles are out-of-scope and will be documented as future enhancements.

### Key Entities *(include if feature involves data)*

- **MapRecord**: Logical representation of an SVG map plus metadata, including file path, tags (e.g., `ci_enabled`), declared spawn zones, and manifest references.
- **VerificationResult**: Structured outcome per map containing status (`pass|fail|warn`), violated rule IDs, remediation hints, duration, and environment instantiation notes.
- **VerificationRunSummary**: Aggregated artifact describing execution metadata (timestamp, git SHA, counts of pass/fail/warn, slow-map list) stored under `output/validation/` and optionally uploaded to dashboards.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: 100% of maps in `maps/svg_maps/` pass the verifier before merges (CI gate ensures ZERO unresolved failures on `main`).
- **SC-002**: Median verification runtime per map ≤ 3 seconds; no single map exceeds the 20-second soft budget or 60-second hard timeout.
- **SC-003**: At least 90% of map-related CI failures include an automated remediation hint, reducing triage time compared to current manual debugging.
- **SC-004**: Documentation (e.g., `docs/SVG_MAP_EDITOR.md`) references the verifier and includes at least one example manifest, ensuring contributors can reproduce the validation workflow.

## Clarifications

### Session 2025-11-20

- Q: How should the verifier instantiate pedestrian-only maps? → A: Use the dedicated pedestrian factory when a map is tagged pedestrian-only, otherwise default to the robot factory.
- Q: Should v1 support remote map bundles? → A: No; limit scope to local repository assets and treat remote bundles as future work.
