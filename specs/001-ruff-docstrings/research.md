# Research Notes: Ruff Docstring Enforcement

## Decision 1: Repository-wide Ruff docstring rules
- **Decision**: Enable Ruff rules D100–D107, D417, D419, D102, and D201 for every Python path (library, scripts, tests, fast-pysf bindings, tooling) with targeted ignores only for generated or third-party vendored files.
- **Rationale**: Ensures consistent documentation expectations, matches clarification that scope is the entire repo, and simplifies CI enforcement by avoiding path-specific rule toggles.
- **Alternatives considered**:
  - Enable rules only for `robot_sf/`: rejected because scripts/tests would still ship undocumented helper logic.
  - Maintain per-directory configs: rejected due to maintenance overhead and risk of drift.

## Decision 2: Treat private helpers pragmatically
- **Decision**: Continue exempting explicitly private helpers (leading underscore) and auto-generated code when Ruff identifies them as such, but require intentional `ignore` entries so exemptions remain auditable.
- **Rationale**: Keeps focus on public API quality while preventing noisy lint failures on internal scaffolding or generated bindings; aligns with Principle XI.
- **Alternatives considered**:
  - Enforce docstrings on every function regardless of prefix: rejected because it would create busywork for trivial test fixtures and degrade signal-to-noise.
  - Blanket-ignore entire directories: rejected because it risks missing public helpers embedded there.

## Decision 3: Document docstring style expectations in Quickstart
- **Decision**: Provide contributors with a concise quickstart explaining the required docstring sections (summary, Args, Returns, Raises) and how Ruff enforces them during CI.
- **Rationale**: Reduces onboarding friction and ensures new docstrings follow the expected structure without trial-and-error via lint failures.
- **Alternatives considered**:
  - Rely solely on Ruff error messages: rejected because they lack project-specific nuance (e.g., referencing Loguru context or schema references).
  - Embed guidance only in existing dev guide: rejected since the feature warrants a targeted quickstart for rapid adoption.

## Repository Python Path Inventory
- `robot_sf/` – Core simulation, navigation, rendering, and benchmark logic.
- `fast-pysf/` – Vendored SocialForce bindings (Python + Cython) used by the simulator.
- `scripts/` – Training, tooling, reporting, and helper CLIs that must expose documented entry points.
- `tests/` – Pytest suite spanning unit/integration/visual regressions that import production APIs.
- `examples/` – Tutorial and advanced workflows that exercise public helpers and require docstring coverage for generated docs.

## Sample Docstring Remediation Checklist
Source: `output/issues/docstrings_summary.json` (grouped via `scripts/tools/docstring_report.py`)

1. `robot_sf/ped_npc/ped_robot_force.py` – 5 violations (D103). Add summaries for helper functions describing how forces are applied.
2. `scripts/hparam_opt.py` – 5 violations (D103). Document CLI entry points so automated tuning scripts are discoverable.
3. `tests/unit/benchmark/test_contract_overlay_text.py` – 5 violations (D103). Convert test helpers into documented fixtures to clarify expected overlay output.
4. `examples/advanced/12_social_force_planner_demo.py` – 4 violations (D101/D102). Classes mocking numpy/planner APIs need class + method docstrings describing the simplified physics.
5. `fast-pysf/benchmarks/forces_benchmark.py` – 4 violations (D100/D101). Module/class docstrings should explain the benchmark intent and expected dataset shapes.

Use this ranking to batch fix by directory: each bullet links to a file and hints at the type of docstring content reviewers expect.
