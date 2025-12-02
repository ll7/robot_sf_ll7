# Copilot Instructions

ALWAYS use the official [dev_guide](../docs/dev_guide.md) as the primary reference for development-related tasks.
It is everyones guide on how to use this repository effectively.

## Additional Instructions

- Use scriptable interfaces instead of cli interfaces when possible.
- Make everything reproducible.
- Central point to link new documentation pages is `docs/README.md`.
  - Link new documentation (sub-)pages in the appropriate section.
- For any changes that affect users, update the `CHANGELOG.md` file.
- Source the environment before using python or uv `source .venv/bin/activate`.

## Recent Changes
- 001-ruff-docstrings: Added Python 3.11 (uv-managed virtual environment) + Ruff (docstring rules), Loguru (logging referenced in docstrings), pytest for regression validation

- 270-imitation-report: Added Python 3.11 (existing uv-managed environment)
- 001-map-verification: Added Python 3.11 (uv-managed virtual environment) + `robot_sf.gym_env` factories + unified configs, Loguru logging, SVG parsing utilities already present in `robot_sf.maps`, optional geometry helpers (Shapely)

## Active Technologies
- Python 3.11 (uv-managed virtual environment) + Ruff (docstring rules), Loguru (logging referenced in docstrings), pytest for regression validation (001-ruff-docstrings)
- Not applicable (source-only change) (001-ruff-docstrings)


## Test Failure Evaluation

**When encountering test failures, always evaluate test significance before fixing:**

1. **Verify test value first**: Not all tests are equally important. Ask:

   - Does this test verify a core public contract (factories, schemas, metrics)?
   - Would this failure impact users in production?
   - Is this testing a known regression or critical edge case?
   - Is the test brittle/flaky (timing, display, environmental issues)?
   - Is the same logic already covered by other tests?

2. **Priority classification**:

   - **High** (fix immediately): Core features, user-facing behavior, schema compliance
   - **Medium** (fix within sprint): Known regressions, important edge cases
   - **Low** (consider archiving): Rare scenarios, redundant coverage, no real-world incidents
   - **Flaky** (refactor or remove): Frequently fails without indicating real bugs

3. **Actions based on priority**:

   - High priority: Fix immediately to protect critical invariants
   - Medium priority: Fix or document deferral with tracking issue
   - Low priority: Consider archiving with documented rationale
   - Flaky: Stabilize with retries/mocks if valuable, else remove

4. **Documentation requirements**:
   - Removing tests requires commit message explaining why
   - New tests need docstring: (1) what is verified, (2) why it matters
   - Deferred failures need "test-debt" label and value assessment

**Reference**: See Constitution Principle XIII and dev_guide.md Testing Strategy for complete criteria.
