# Feature Specification: Clean Root Output Directories

**Feature Branch**: `243-clean-output-dirs`  
**Created**: November 13, 2025  
**Status**: Draft  
**Input**: User description: "Clean up root-level output directory clutter"

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

### User Story 1 - Maintainer sees clean root (Priority: P1)

A maintainer clones the repository or pulls latest changes, runs the standard validation tasks, and can immediately understand that all generated artifacts live under a single designated location instead of being scattered across the root directory.

**Why this priority**: A clean root directory is the primary improvement promised by the issue and directly impacts day-to-day workflows for every contributor.

**Independent Test**: After running the documented validation tasks, list the repository root and verify that only source/config directories plus a single designated artifact root are present.

**Acceptance Scenarios**:

1. **Given** a fresh checkout on main, **When** the maintainer runs `uv run pytest tests`, **Then** no new artifact directories appear at the repository root outside the approved list.
2. **Given** a repository with existing artifact directories in legacy locations, **When** the maintainer runs the migration workflow, **Then** the root directory no longer contains `results/`, `recordings/`, `htmlcov/`, or other legacy artifact folders.

---

### User Story 2 - CI artifacts stay organized (Priority: P2)

The automation owner runs CI or scheduled benchmarks and confirms that generated coverage reports, benchmark JSON files, recordings, and wandb artifacts are stored in predictable subdirectories that can be archived or cleaned in bulk.

**Why this priority**: CI and nightly jobs produce the majority of artifacts; keeping them organized prevents regressions and simplifies maintenance workloads.

**Independent Test**: Trigger a CI-equivalent run locally (e.g., execute coverage, benchmark, and recording scripts) and confirm all artifacts land in the standardized structure with no stray directories created.

**Acceptance Scenarios**:

1. **Given** the benchmark and coverage scripts configured with default paths, **When** the automation owner executes them, **Then** all artifacts resolve under the approved artifact root in their assigned category directories.
2. **Given** an attempt to create a new top-level directory during CI, **When** the guard check runs, **Then** it fails the run with a descriptive message pointing to the artifact policy.

---

### User Story 3 - Documentation highlights artifact policy (Priority: P3)

A new contributor reads the onboarding and validation docs and learns where artifacts are stored, how to clean them up, and how to override the location for experiments without digging through source code.

**Why this priority**: Clear documentation reinforces the change and reduces support questions, but it relies on the structural work from the higher-priority stories.

**Independent Test**: Review the updated documentation and confirm it references the standardized artifact root, includes cleanup guidance, and no longer lists legacy directories.

**Acceptance Scenarios**:

1. **Given** the contributor reads `docs/dev_guide.md` or the main README, **When** they follow the instructions to inspect generated artifacts, **Then** the docs direct them to the standardized location and the instructions match observed behavior.
2. **Given** a contributor wants to override artifact locations, **When** they follow the documented guidance (e.g., environment variable or config flag), **Then** their custom path works without leaving stray directories in the repository root.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

- What happens when legacy directories (e.g., `results/`, `recordings/`, `htmlcov/`) already exist with large contents and the migration aborts midway? The process must be resumable without data loss.
- How does the system handle contributors who have `ROBOT_SF_ARTIFACT_ROOT` set to a custom path? The default restructuring must respect overrides while keeping the root clean.
- What occurs if a script hardcodes a legacy path and cannot be migrated immediately? A fallback or explicit failure mode must be defined.
- How are git-ignored files such as `benchmark_results.json` treated if they are meant to be user-facing exports versus throwaway artifacts?

## Assumptions

- Default local contributor workflows rely on the documented quality gates (`uv run pytest tests`, coverage, benchmarks) as representative artifact producers.
- The existing `ROBOT_SF_ARTIFACT_ROOT` environment variable remains the supported override mechanism for redirecting artifacts outside the repository tree.
- Contributors are comfortable running a migration helper script as part of the change announcement to consolidate any lingering directories.
- The canonical artifact destination will live under a single top-level `output/` directory with clearly named subdirectories per artifact family.
- Core artifact producers will be updated ahead of enforcement so that fail-fast checks surface only unexpected residual usages of legacy paths.

## Clarifications

### Session 2025-11-13

- Q: Should the canonical artifact structure use one `output/` directory or split between `.cache/` and `results/`? → A: Use a single `output/` root with subdirectories per artifact type.
- Q: How strictly should legacy artifact paths be handled after restructuring? → A: Fail fast with descriptive errors once most expected producers are updated.

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: The project MUST define a canonical artifact root that replaces the current scatter of `results/`, `recordings/`, `htmlcov/`, `tmp/`, `wandb/`, `benchmark_results.json`, and `coverage.json` in the repository root.
- **FR-002**: The canonical artifact root MUST be a single `output/` directory that organizes artifacts into stable subdirectories grouped by purpose (coverage, benchmarks, recordings, temporary files).
- **FR-003**: A migration workflow MUST relocate all existing root-level artifact directories into the canonical structure and emit a summary report of moved items so contributors know what changed.
- **FR-004**: All scripts, validation harnesses, and documentation MUST respect the artifact root default while still honoring `ROBOT_SF_ARTIFACT_ROOT` overrides, including automated tests that run under the current environment variable-based rerouting.
- **FR-005**: The documentation set (README, `docs/dev_guide.md`, relevant scripts) MUST be updated to explain the new artifact policy, including cleanup guidance and override instructions.
- **FR-006**: After proactively updating the known artifact producers, the project MUST fail fast with descriptive errors whenever a process attempts to write to legacy root-level paths, guiding contributors to the `output/` hierarchy.
- **FR-007**: A guard check MUST exist (local script or CI step) that fails when unapproved top-level artifact directories or files are introduced, providing actionable guidance to contributors.

### Key Entities *(include if feature involves data)*

- **Artifact Root Policy**: Describes the canonical directory layout (default root, mandatory subdirectories, allowed transient locations) and is referenced by scripts and documentation.
- **Migration Report**: A machine-readable summary (e.g., JSON or table) enumerating which legacy directories/files were moved, skipped, or required manual intervention during migration.
- **Artifact Producer**: Any script or automated process that generates files; each producer must declare its target subdirectory in the policy so that guard checks can validate compliance.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: After running the documented validation sequence on a clean checkout, the repository root contains no unapproved artifact directories or files; compliance is verified by an automated guard that reports zero violations.
- **SC-002**: The migration workflow relocates 100% of the tracked legacy artifact directories (`results/`, `recordings/`, `htmlcov/`, `tmp/`, `wandb/`, `benchmark_results.json`, `coverage.json`) into the new structure in a single run, leaving only `.gitignore` placeholders (if any) behind.
- **SC-003**: Updated documentation explicitly references the artifact policy and is validated by at least one onboarding walkthrough that confirms the instructions match observed artifact locations.
- **SC-004**: All CI pipelines and nightly jobs complete without path-related failures, and their published artifacts reside exclusively inside the canonical structure as confirmed by build logs or artifact manifests.
