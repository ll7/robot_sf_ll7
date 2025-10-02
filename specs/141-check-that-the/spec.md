# Feature Specification: Verify Feature Extractor Training Flow

**Feature Branch**: `141-check-that-the`  
**Created**: 2025-10-02  
**Status**: Draft  
**Input**: User description: "Check that the new feature extractor works as intended in scripts/multi_extractor_training.py. Modify output to ./tmp, ensure single-thread fallback and cross-platform (mac m4, Ubuntu RTX) configs. Optionally refactor feature extractor helpers for reuse."

---

## 1. Goals

### Primary User Story
A research engineer wants to compare multiple feature extractors for social-navigation training runs. They launch the provided training script, expect runs to complete reliably on both macOS (Apple Silicon) and Ubuntu with NVIDIA GPUs, and receive organized outputs under a temporary workspace for analysis.

### Secondary Scenarios
- The engineer enables a high-performance mode when running on an Ubuntu RTX workstation to speed up experiments while keeping the workflow reproducible.
- Another teammate inspects earlier runs and needs human-readable metadata to understand which hardware and execution mode produced each result.

### Acceptance Scenarios
1. **Given** the engineer runs the training script with default settings on macOS M4 hardware, **When** the script initializes environments in single-thread mode, **Then** each feature extractor completes training epochs and saves metrics and checkpoints under `./tmp/...` without crashes.
2. **Given** the engineer runs the training script on Ubuntu with an NVIDIA RTX GPU, **When** they enable the documented multi-process configuration, **Then** the script executes parallel environments successfully while writing outputs to the temporary results directory and noting the hardware profile in the summary.
3. **Given** a feature extractor configuration fails validation, **When** the script evaluates the configuration list, **Then** the failure is logged, the run is skipped, and the remaining extractors continue without terminating the workflow.

### Edge Cases
- What happens when a feature extractor configuration is missing required parameters? The script should skip the run and log the missing configuration while continuing with other extractors.
- How does the system handle hardware without GPU acceleration? Training should fall back to CPU single-thread mode with a clear warning about expected performance impacts.
- What if the temporary directory already contains outputs from a previous run? The workflow should either create versioned subdirectories or surface guidance on clearing stale artifacts.

### Non-Goals
- Optimizing multi-threaded training performance beyond establishing a working configuration on Ubuntu RTX hardware.
- Redesigning the feature extractor architectures themselves; only integration and verification are in scope.

---

## 2. Requirements

### Functional Requirements
- **FR-001**: The training workflow MUST validate that each supported feature extractor can be launched and trained using the comparison script without runtime crashes.
- **FR-002**: The script MUST default to single-thread execution and document the steps required to switch to high-performance (multi-process or GPU-accelerated) mode.
- **FR-003**: Training outputs MUST be written to a temporary results directory under `./tmp/…` with per-extractor subfolders for checkpoints, logs, and metrics summaries.
- **FR-004**: The comparison summary MUST highlight hardware configuration differences (macOS M4 vs. Ubuntu RTX) so that users understand expected performance variance.
- **FR-005**: Reusable helper logic for feature extractor configuration and results handling MUST be extracted into shared modules when it benefits other areas of the repository.
- **FR-006**: The workflow MUST produce user-facing guidance for handling missing or failing extractor runs, including how the script continues when a configuration cannot initialize.
- **FR-007**: The system MUST record training metadata sufficient for cross-platform reproducibility (e.g., timestamp, hardware profile, single vs. multi-process mode).
- **FR-008**: [NEEDS CLARIFICATION: Are there minimum benchmark metrics or convergence thresholds that each extractor must reach?]

### Non-Functional Requirements
- **NFR-001**: The default single-thread run MUST complete without crashes on macOS M4 hardware using the spawn start method.
- **NFR-002**: High-performance mode MUST be operable on Ubuntu RTX hardware without manual code edits, relying on documented configuration toggles.
- **NFR-003**: Output artifacts MUST be organized to avoid cluttering the repository and facilitate cleanup (e.g., versioned folders under `./tmp`).
- **NFR-004**: Logging MUST clearly differentiate between validation warnings and fatal errors to guide user action.

### Assumptions & Dependencies
- macOS environments use Apple Silicon (M4) and require spawn-based multiprocessing with OBJC fork safety enabled.
- Ubuntu RTX environments have access to CUDA and can optionally run multiple parallel environments; exact worker counts depend on available GPU memory.
- Existing feature extractor implementations remain unchanged; only integration and orchestration adjustments are anticipated.
- The repository’s documentation framework (docs/dev/issues/…) will capture the implementation plan and checklist for this feature.
- Dependencies on Stable-Baselines3, Gymnasium, and current training utilities remain valid; no major version upgrades are scoped.
- Output path `./tmp` is acceptable for intermediate artifacts and is not tracked by version control.

### Key Entities
- **Training Run Summary**: Represents the aggregated results for each feature extractor, including metadata (timestamp, hardware mode, run status) and key performance metrics.
- **Extractor Configuration Profile**: Describes the logical parameters selected for each feature extractor comparison (name, parameter set, expected resources) without implementation specifics.

---

## 3. Open Questions
1. **Benchmark expectations**: Are there minimum convergence metrics or training duration targets that must be met before a run is considered successful? *(Blocks FR-008)*
2. **Tmp directory policy**: Should previous artifacts under `./tmp` be automatically cleaned before new runs, or should the workflow retain history for comparison?
3. **Result reporting format**: Do stakeholders require a consolidated report (e.g., JSON, Markdown table) summarizing all extractor runs, or is per-run metadata sufficient?

---

## 4. Review Checklist
- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed (pending resolution of open questions)

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [ ] Dependencies and assumptions identified (pending confirmation of tmp directory policy and benchmark expectations)

---

## 5. Approval
- **Product/Stakeholder**: _TBD_
- **Engineering Lead**: _TBD_
- **QA/Testing**: _TBD_
