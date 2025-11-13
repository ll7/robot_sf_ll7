# Data Model: Organize and Categorize Example Files

## Entity: ExampleCategory
- **Description**: Logical tier representing a learning path stage for example scripts.
- **Fields**:
  - `slug` (string, required): Lowercase directory name (`quickstart`, `advanced`, `benchmarks`, `plotting`).
  - `title` (string, required): Human-friendly category name displayed in documentation.
  - `description` (string, required): Short explanation of learning goal/who should use the category.
  - `order` (integer, required): Ordering index for README and nav tables.
  - `ci_default` (boolean, required, default `true`): Whether scripts in this category are expected to run in CI by default.
- **Relationships**:
  - 1-to-many with `ExampleScript` (category owns multiple scripts).
- **Validation Rules**:
  - `slug` must match directory names exactly.
  - `order` must be unique per category to avoid ambiguous sorting.
  - `ci_default` can be overridden per script but defaults to category value.

## Entity: ExampleScript
- **Description**: Individual example Python entry point demonstrating a workflow or capability.
- **Fields**:
  - `path` (string, required): Relative file path under `examples/` (e.g., `quickstart/01_basic_robot.py`).
  - `name` (string, required): Display name used in README index.
  - `summary` (string, required): One-line purpose statement (mirrors docstring first sentence).
  - `category_slug` (string, required): Foreign key referencing `ExampleCategory.slug`.
  - `prerequisites` (list[string], optional): External assets or models required.
  - `ci_enabled` (boolean, required): Whether CI harness must execute this script.
  - `ci_reason` (string, required when `ci_enabled` is `false`): Justification recorded in manifest and README.
  - `doc_reference` (string, optional): Documentation path referencing this example (e.g., `docs/ENVIRONMENT.md#quickstart`).
  - `tags` (list[string], optional): Feature flags (e.g., `image`, `benchmark`, `visualization`).
- **Relationships**:
  - Belongs to exactly one `ExampleCategory`.
  - Linked to zero or more documentation sections (soft reference via `doc_reference`).
- **Validation Rules**:
  - File must exist and contain a top-level docstring matching `summary` sentence.
  - `ci_enabled` defaults to `ExampleCategory.ci_default` unless explicitly overridden.
  - Examples in `_archived/` category must have `ci_enabled = false` with `ci_reason`.

## Entity: ExampleManifest
- **Description**: Machine-readable manifest describing categories and scripts for automation (documentation index, CI harness).
- **Fields**:
  - `version` (string, required): Semantic version for manifest schema (`1.0.0` initial).
  - `categories` (list[`ExampleCategory`], required): Category definitions.
  - `examples` (list[`ExampleScript`], required): Example entries linked to categories.
- **Relationships**:
  - Aggregates categories and scripts; consumed by README generator and smoke harness.
- **Validation Rules**:
  - All `ExampleScript.category_slug` values must exist in `categories`.
  - Paths must remain within `examples/` tree and avoid `_archived/` unless flagged.
  - Manifest must be kept in sync with directory structure during CI (validation check fails otherwise).

## Entity: ExampleDocstring
- **Description**: Structured content at the top of each script describing usage.
- **Fields**:
  - `purpose` (string, required): First sentence summarizing example intent.
  - `usage` (string, required): Code block or command snippet demonstrating execution.
  - `prerequisites` (list[string], optional): Mirrors `ExampleScript.prerequisites`.
  - `expected_output` (string, optional): Description of results (e.g., "renders path animation to stdout").
  - `limitations` (string, optional): Known constraints or headless mode adjustments.
- **Validation Rules**:
  - Docstring must be present and parseable (triple-quoted string at module top).
  - `purpose` line must match `ExampleScript.summary` for consistency.
  - Usage block should reference reproducible commands (prefer `uv run python ...`).

## Entity: ArchivedExampleRecord
- **Description**: Documentation artifact for examples moved to `_archived/`.
- **Fields**:
  - `path` (string, required): Relative path under `_archived/`.
  - `replacement` (string, required): Path/name of canonical replacement example.
  - `reason` (string, required): Explanation (e.g., "requires manual interaction").
  - `last_validated` (date, optional): Last time the example was confirmed functional.
- **Validation Rules**:
  - Every archived example must include a record entry and README note.
  - `replacement` must point to an active example unless functionality removed entirely (documented as `replacement: null`).
