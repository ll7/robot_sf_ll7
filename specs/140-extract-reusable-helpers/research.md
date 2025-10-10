# Phase 0 Research: Reusable Helper Consolidation

## Research Goals
- Build a repeatable inventory process for identifying reusable helper logic across `examples/` and `scripts/`.
- Define criteria for when extraction into `robot_sf/` is mandatory versus optional.
- Map helper categories to existing library packages to avoid scattering utilities.
- Confirm validation coverage needed to guarantee behavior parity post-extraction.

## Findings

### Inventory Methodology
- **Decision**: Use a two-pass static audit: (1) keyword scan for repeated helper names/patterns (env setup, recording, benchmark loops), (2) manual review of high-usage demos (`classic_interactions`, `demo_*`, `run_social_navigation_benchmark.py`).
- **Rationale**: Balances speed with accuracy—automated scan surfaces candidates; manual pass confirms true reusability.
- **Alternatives Considered**: AST-based diff tooling (excess complexity) and manual-only audit (risk of omissions).

### Extraction Criteria
- **Decision**: Extract helpers when they meet any of: used in ≥2 maintained scripts, exceed 15 LOC with branching, or manage shared resources (env creation, recording, logging setup).
- **Rationale**: Aligns with Constitution Principle XI (library reuse) while avoiding churn for trivial wrappers.
- **Alternatives Considered**: Blanket extraction of all helpers (unnecessary churn) or only multi-use helpers (misses single-use but complex utilities).

### Target Library Placement
- **Decision**: Place helpers near existing domain modules:
  - Environment setup → `robot_sf.gym_env` or `robot_sf.benchmark` utilities.
  - Recording/rendering → `robot_sf.render`.
  - Benchmark orchestration → `robot_sf.benchmark.utils` or submodule `orchestration` (to be added if needed).
- **Rationale**: Keeps helper responsibilities aligned with current package boundaries for discoverability.
- **Alternatives Considered**: Create a new `robot_sf.helpers` package (risks duplicating existing structure).

### Validation & Regression Coverage
- **Decision**: Require the following to pass before/after extraction:
  - `uv run pytest tests`
  - `uv run pytest test_pygame` (headless rendering smoke)
  - `scripts/validation/test_basic_environment.sh`
  - `scripts/validation/test_complete_simulation.sh`
- **Rationale**: These cover core env behavior, rendering, and end-to-end simulation.
- **Alternatives Considered**: Full benchmark runs (too slow for iterative refactor) or minimal unit tests only (insufficient coverage).

## Open Questions Resolved
- Clarified scope exclusion for one-off validation/debug scripts (handled at specification stage; no additional research needed).

## Next Steps
- Proceed to Phase 1 design using the inventory criteria and placement decisions above.
