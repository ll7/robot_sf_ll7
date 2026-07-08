# Feature Specification: Organize and Categorize Example Files

**Feature Branch**: `245-organize-examples`  
**Created**: November 12, 2025  
**Status**: In Progress  
**Input**: User description: "Reorganize and categorize 84 example files into tiered structure with clear documentation for improved discoverability and maintainability"

## Overview

The `examples/` directory currently contains 84 files with overlapping content, unclear purpose, and no organizational structure. Users struggle to identify which examples to study, and maintainers find it difficult to detect obsolete or broken examples. This feature creates a clear tiered organization with comprehensive documentation to guide users through increasingly complex use cases while maintaining example validity.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - New User Discovers Starting Point (Priority: P1)

A first-time user downloads the repository and wants to understand how to use robot-sf. They need to find a clear entry point that demonstrates basic functionality without overwhelming complexity.

**Why this priority**: Onboarding is the critical first experience. Without a clear entry point, users may abandon the project or use outdated examples.

**Independent Test**: Can be fully tested by navigating `examples/README.md`, following the quick-start section, and successfully running a basic example end-to-end.

**Acceptance Scenarios**:

1. **Given** a new user opens `examples/README.md`, **When** they read the quick-start guide, **Then** they can identify and run the first recommended example within 2 minutes
2. **Given** a new user wants to understand available use cases, **When** they view the decision tree in README, **Then** they can determine which example best matches their needs
3. **Given** a new user runs a quickstart example, **When** they execute it, **Then** it completes without errors and demonstrates core functionality

---

### User Story 2 - Developer Finds Feature-Specific Examples (Priority: P1)

A developer wants to implement a specific feature (e.g., image observations, custom reward functions) and needs targeted examples showing how to do this.

**Why this priority**: Examples are primary learning resources for feature adoption. Without organized feature examples, users may miss important capabilities.

**Independent Test**: Can be fully tested by searching the examples directory structure, locating a feature-specific example, and understanding how to use it from its docstring and structure.

**Acceptance Scenarios**:

1. **Given** a developer looks for "image observations" example, **When** they check the advanced directory, **Then** they find a clearly-named, documented example demonstrating this feature
2. **Given** an example is chosen, **When** the developer reads its docstring, **Then** they understand the example's purpose, dependencies, and how to run it without external documentation
3. **Given** a developer wants to extend an example, **When** they examine the code structure, **Then** clear comments indicate extension points and assumptions

---

### User Story 3 - Maintainer Identifies Obsolete Examples (Priority: P1)

A repository maintainer needs to regularly audit which examples are current, which are deprecated, and which might be broken. They need confidence that highlighted examples work with the current codebase.

**Why this priority**: Stale examples mislead users and create maintenance burden. Maintainers need clear visibility into example status.

**Independent Test**: Can be fully tested by examining the `_archived/` directory structure, comparing it to active examples, and verifying that deprecated examples contain clear migration guidance.

**Acceptance Scenarios**:

1. **Given** an example is obsolete, **When** it's moved to `_archived/`, **Then** it includes a clear note explaining why it's archived and which current example to use instead
2. **Given** a maintainer reviews active examples, **When** they scan example directories, **Then** all examples in active directories have clear, recent docstrings with purpose and status
3. **Given** a CI/test system runs, **When** all examples in active directories are executed, **Then** they complete without errors or import failures

---

### User Story 4 - Documentation Reader Finds Visual and Complex Examples (Priority: P2)

A user reading the documentation wants to see visualizations or run complex benchmark scenarios referenced in docs. They need to locate plotting and benchmarking examples.

**Why this priority**: Visualization and benchmarking are secondary but important workflows. Organization enables documentation to reliably reference examples.

**Independent Test**: Can be fully tested by finding plotting/benchmark examples through the directory structure and executing them to produce expected outputs.

**Acceptance Scenarios**:

1. **Given** documentation mentions "Pareto frontier visualization," **When** a user browses `examples/plotting/`, **Then** they find a clearly-named example demonstrating this
2. **Given** a user wants to run a benchmark scenario, **When** they check `examples/benchmarks/`, **Then** they find organized, documented benchmark runners
3. **Given** an example produces outputs (plots, data), **When** it executes, **Then** outputs are saved to predictable locations documented in the example's docstring

---

### Edge Cases

- **Duplicate examples**: When multiple examples implement similar functionality (e.g., `demo_pedestrian.py` vs `demo_pedestrian_updated.py`), the newer/canonical version remains active while the older is archived with clear migration notes.
- **Interdependent examples**: When one example depends on outputs from another (e.g., benchmarking requires trained models), the docstring clearly documents prerequisites and build order.
- **Changed APIs**: If repository APIs change and an example becomes broken, the example is either updated immediately (if in active directories) or moved to `_archived/` with explanation of required API changes.
- **Example variations**: When multiple legitimate variations of an example exist (e.g., with/without GPU, different backends), they are documented together in the same subdirectory with clear naming conventions.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST organize examples into tiered directories: `quickstart/`, `advanced/`, `benchmarks/`, `plotting/`, and `_archived/`, each serving a distinct user need
- **FR-002**: System MUST include a comprehensive `examples/README.md` containing:
  - Quick-start guide explaining how to run first example
  - Decision tree helping users choose appropriate example by use case
  - Purpose and contents of each directory
  - Index or table of all examples with one-line descriptions
  - Prerequisites and environment setup instructions
- **FR-003**: Each example file MUST include a docstring header explaining:
  - Purpose of the example and what it demonstrates
  - Prerequisites (dependencies, trained models, data files)
  - How to run the example and expected output format
  - Where in the documentation this example is referenced (if applicable)
  - Any limitations or assumptions
- **FR-004**: System MUST clearly identify deprecated/archived examples:
  - Archived examples moved to `examples/_archived/`
  - Each archived example includes a migration note explaining why it's archived and which current example to use instead
  - Archived directory contains a README explaining the archival policy
- **FR-005**: System MUST maintain or migrate import paths:
  - All examples in active directories import from robot-sf using current, stable import paths
  - If directory reorganization affects imports within examples, imports are updated to reflect any structural changes
- **FR-006**: System MUST ensure example maintainability:
  - Each example directory (quickstart, advanced, etc.) is small enough that one engineer can validate all examples run correctly in under 1 hour
  - Example selection and categorization follows clear criteria documented in README

### Key Entities

- **Example**: A standalone Python script demonstrating a specific feature, workflow, or use case; includes code, docstring, and configuration
- **Example Directory**: Thematic grouping of examples (quickstart, advanced, benchmarks, plotting) organized by user journey and complexity
- **Docstring Header**: Structured documentation at the top of each example file including purpose, prerequisites, usage, and limitations
- **Migration Note**: Text explaining why an example is archived and which current example provides equivalent or superior functionality
- **Decision Tree**: User-friendly guide in README helping select appropriate examples based on use case or learning goal

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 84 example files are categorized—either organized into active directories or moved to `_archived/` with migration notes—100% coverage
- **SC-002**: Each active example file includes a docstring header documenting purpose, prerequisites, usage, and limitations; verification: grep for required docstring fields in all examples
- **SC-003**: `examples/README.md` is created with decision tree, quick-start guide, directory descriptions, and index; verified by manual review and 100% example coverage in index
- **SC-004**: Deprecated examples (those with "updated" variants) are moved to `_archived/` with clear migration guidance; target: all duplicates consolidated, users directed to canonical versions
- **SC-005**: CI pipeline includes a check that all examples in active directories execute without import errors or runtime exceptions; target: 100% of active examples pass execution check
- **SC-006**: New users can follow the quick-start guide to run their first example within 5 minutes without external documentation; verified by user testing or self-guided walkthrough
- **SC-007**: Documentation references to examples are updated to use new directory structure; target: all broken links or stale references in `docs/` and main README resolved
- **SC-008**: Example organization reduces support/issue burden related to "which example should I use?" by making selection unambiguous; measured by reduction in related issues post-release

## Assumptions

1. **Reasonable defaults applied** (examples are tools for learning, not critical functionality):
   - Quickstart directory targets absolute beginners; 3–5 examples showing basic environment, model loading, and custom maps
   - Advanced directory focuses on feature-specific use cases; ~10–15 examples covering backend selection, feature extractors, multi-robot, image observations, pedestrian environments
   - Benchmarks and plotting directories provide evaluation/visualization workflows; ~5–8 each
   - Archived examples retain full code but are hidden from primary user paths

2. **Criteria for moving examples to archive**:
   - Duplicate functionality with a newer/canonical version available
   - Deprecated or experimental features no longer supported in current codebase
   - Broken due to API changes and lower-priority than fixing (lower-demand features)
   - Cannot be archived simply by docstring if functionality is genuinely needed; in that case, example is updated instead

3. **Import paths and API stability**:
   - Examples use factory pattern (`make_robot_env()`, etc.) and stable public APIs from `robot_sf` package
   - Examples are not expected to import from internal/private modules or subtrees directly
   - If examples are moved to subdirectories, imports remain via public package APIs; no path manipulation needed

4. **Maintenance scope**:
   - This feature addresses organization and documentation; it does not rewrite or refactor example code logic
   - Example code correctness is verified by CI; logic improvements are out of scope and addressed separately
   - Archived examples are not actively maintained; they remain as-is with migration notes

5. **Documentation integration**:
   - Main README.md will link to `examples/README.md` as the entry point for learning examples
   - Existing documentation pages that reference specific examples are updated to use new directory structure
   - New examples or category additions follow the established tier-based structure

## Related Issues & Context

- **Upstream issue**: Repository structure analysis notes user confusion about example organization
- **User feedback**: Multiple support requests show uncertainty about which example to start with
- **Maintenance burden**: Test suite may pass but examples can fail silently; organization enables per-category validation
- **Documentation integration**: Documentation references examples but structure is not evident to users navigating directly to `examples/` directory
