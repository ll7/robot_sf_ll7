# Data Model: Ruff Docstring Enforcement

## Entities

### DocstringRuleSet
- **Description**: Configuration bundle that defines which Ruff docstring rules are enabled and how they apply across repository paths.
- **Fields**:
  - `name` (string) – human-readable identifier (e.g., "docstrings_full_repo").
  - `rules` (list[string]) – ordered list of rule codes (D100–D107, D417, D419, D102, D201).
  - `include_paths` (list[path]) – glob patterns covered by enforcement.
  - `exclude_paths` (list[path]) – explicit exemptions (generated/vendor code) with justification links.
  - `severity` (enum: warning|error) – gating level; must be `error` for CI per requirements.
- **Relationships**: Referenced by `LintWorkflow` to determine gating behavior.

### LintWorkflow
- **Description**: Describes how linting executes locally and in CI to enforce docstring compliance.
- **Fields**:
  - `command` (string) – canonical invocation (`uv run ruff check`).
  - `environment` (string) – environment activation instructions (`source .venv/bin/activate`).
  - `ci_job` (string) – name/identifier of CI task running the command.
  - `blocking` (bool) – indicates job must pass before merge (true).
  - `artifacts` (list[path]) – optional lint reports or logs stored for debugging.
- **Relationships**: Uses `DocstringRuleSet` for configuration and outputs `LintFindings`.

### LintFinding
- **Description**: Individual docstring violation or confirmation produced by Ruff.
- **Fields**:
  - `file_path` (path) – path to offending file.
  - `line` (int) – starting line of violation.
  - `rule` (string) – docstring rule identifier.
  - `message` (string) – Ruff-provided remediation hint.
  - `status` (enum: open|fixed) – tracked during remediation sweep.
- **Relationships**: Aggregated under `LintWorkflow` runs; resolution tracked in tasks.

### ContributorGuide
- **Description**: Documentation artifact (quickstart) that teaches contributors the enforced docstring style.
- **Fields**:
  - `location` (path) – quickstart path (`specs/001-ruff-docstrings/quickstart.md`).
  - `sections` (list[string]) – summary, how-to-run, troubleshooting.
  - `last_updated` (date) – sync with plan.
- **Relationships**: Cited by onboarding docs and referenced in PR templates if needed.
