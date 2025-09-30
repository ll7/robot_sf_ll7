# Phase 1 Data Model: Reusable Helper Catalog

## Entities

### HelperCategory
- **Purpose**: Groups related helper capabilities (e.g., environment_setup, recording, benchmarking, analysis).
- **Attributes**:
  - `key` (str, unique, snake_case)
  - `description` (str)
  - `target_module` (python import path, e.g., `robot_sf.benchmark.utils`)
  - `default_owner` (maintainer or team alias)
- **Relationships**: One-to-many with `HelperCapability`.

### HelperCapability
- **Purpose**: Describes a reusable helper function or class.
- **Attributes**:
  - `name` (str, unique within target module)
  - `category_key` (foreign key to HelperCategory)
  - `summary` (str, single sentence docstring requirement)
  - `inputs` (list of parameters with type hints)
  - `outputs` (return type / side effects)
  - `dependencies` (list of external modules relied upon)
  - `tests` (path to validating test/validation script)
  - `docs_link` (URL or relative doc path)
- **Relationships**: Many-to-many with `ExampleOrchestrator` via `OrchestratorUsage` linking table.

### ExampleOrchestrator
- **Purpose**: Represents an example or script that should act purely as an orchestrator.
- **Attributes**:
  - `path` (filesystem path under `examples/` or `scripts/`)
  - `owner` (maintainer alias)
  - `requires_recording` (bool)
  - `notes` (free-form text for deviations)
- **Relationships**: Many-to-many with `HelperCapability` via `OrchestratorUsage`.

### OrchestratorUsage (linking record)
- **Purpose**: Tracks which orchestrators consume which helpers.
- **Attributes**:
  - `orchestrator_path`
  - `helper_name`
  - `integration_notes` (e.g., configuration overrides)

### RegressionCheck
- **Purpose**: Defines the validation commands required to prove parity.
- **Attributes**:
  - `command` (string command to run)
  - `description`
  - `frequency` (e.g., pre-commit, nightly)

## State & Lifecycle
- HelperCapabilities move through states: `identified` → `extracted` → `documented` → `validated`.
- Promotion to `extracted` requires library module creation with docstring.
- Promotion to `documented` requires docs link entry.
- Promotion to `validated` requires associated tests/validation scripts recorded in `RegressionCheck`.

## Data Quality Rules
- Every HelperCapability MUST belong to exactly one HelperCategory.
- Each ExampleOrchestrator MUST reference at least one HelperCapability after refactor.
- RegressionCheck list MUST cover each HelperCategory (ensuring broad coverage).
- Docs link MUST resolve to `docs/` or `examples/README.md` entry for traceability.
