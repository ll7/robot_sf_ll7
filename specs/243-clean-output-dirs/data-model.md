# Data Model: Clean Root Output Directories

**Feature Branch**: `243-clean-output-dirs`
**Date**: November 13, 2025

## Entities

### ArtifactRoot
- **Description**: Canonical directory (`output/`) containing all generated artifacts.
- **Attributes**:
  - `path` (string): Absolute path to the artifact root; defaults to `<repo_root>/output`.
  - `subdirectories` (list[str]): Required child directories (e.g., `coverage/`, `benchmarks/`, `recordings/`, `wandb/`, `tmp/`).
  - `overrides_enabled` (bool): Indicates whether `ROBOT_SF_ARTIFACT_ROOT` overrides the default root.
- **Relationships**:
  - Owns multiple `ArtifactCategory` entries.
- **Lifecycle**:
  - Created during repository setup or migration script execution.
  - Updated if new categories are added via configuration updates.

### ArtifactCategory
- **Description**: Represents a logical grouping of artifacts underneath `output/`.
- **Attributes**:
  - `name` (string): Category identifier (e.g., `coverage`).
  - `purpose` (string): Human-readable description (e.g., "Stores pytest coverage reports").
  - `retention_hint` (enum[str]): Suggested cleanup strategy (`short-lived`, `keep-latest`, `long-lived`).
  - `producers` (list[str]): Scripts or workflows expected to write into the category.
- **Relationships**:
  - Belongs to one `ArtifactRoot`.

### ArtifactProducer
- **Description**: Script or workflow that generates artifacts.
- **Attributes**:
  - `name` (string): Identifier for the producer (e.g., `uv run pytest tests`).
  - `outputs` (list[str]): Target `ArtifactCategory` names.
  - `guard_enforced` (bool): Whether new guard check covers this producer.
  - `migration_status` (enum[str]): `pending`, `updated`, or `verified`.
- **Relationships**:
  - References one or more `ArtifactCategory` entries.

### LegacyPathViolation
- **Description**: Captures attempts to write to deprecated root-level paths.
- **Attributes**:
  - `path` (string): Legacy directory detected.
  - `producer` (string): Process or script responsible for the violation.
  - `timestamp` (datetime): Detection time.
  - `remediation` (string): Suggested fix or migration note.
- **Relationships**:
  - Reported by guard checks; optionally linked back to `ArtifactProducer`.

## Validation Rules
- `ArtifactRoot.subdirectories` must always include the baseline categories: `coverage`, `benchmarks`, `recordings`, `wandb`, `tmp` (extensible via configuration).
- `ArtifactCategory.producers` must align with documented scripts; guard checks fail if an unknown producer writes to a category.
- `ArtifactProducer.outputs` must reference valid category names; invalid references are flagged during configuration validation.
- `LegacyPathViolation.path` must belong to the known legacy set (`results`, `recordings`, `wandb`, `tmp`, `htmlcov`, `benchmark_results.json`, `coverage.json`).

## State Transitions
- `ArtifactProducer.migration_status`: `pending` → `updated` once scripts adopt the new root; `updated` → `verified` after guard checks confirm no legacy writes for that producer across CI runs.
- `LegacyPathViolation` entries are generated when guard checks trip; resolution moves the producer toward `verified` status.
