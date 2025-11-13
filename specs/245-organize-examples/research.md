# Research Notes: Organize and Categorize Example Files

## 1. Example Execution Harness Integration
- **Decision**: Implement a pytest-based smoke module (`tests/examples/test_examples_run.py`) that discovers active example scripts and executes each via `subprocess.run` in headless mode.
- **Rationale**: The repository already relies on pytest for unified CI (see `docs/dev_guide.md` quality gates and existing smoke tests in `tests/test_runner_smoke.py`), so adding a parametrized test keeps tooling consistent, benefits from pytest fixtures (temp dirs, environment control), and fits naturally into existing CI without extra job wiring.
- **Alternatives considered**: (a) Standalone shell script similar to `scripts/validation/test_basic_environment.sh` — rejected because it would duplicate command orchestration and complicate failure reporting across platforms. (b) Custom CLI runner inside `scripts/validation/` — rejected since pytest already offers parallelism, reporting, and skip markers.

## 2. Performance Budget & Coverage Strategy
- **Decision**: Run all scripts in `examples/quickstart/`, `examples/advanced/`, `examples/benchmarks/`, and `examples/plotting/` by default, with per-script metadata allowing explicit opt-out (e.g., `ci_skip: true` with documented justification) to keep runtime under 10 minutes.
- **Rationale**: Success Criterion SC-005 requires a CI check covering all active examples. Empirical review shows some scripts (e.g., `examples/interactive_playback_demo.py`) require manual interaction or large assets; providing a manifest (`examples/examples_manifest.yaml`) lets us annotate such cases while keeping the default contract “run everything”. Metadata also aids documentation and maintenance audits.
- **Alternatives considered**: (a) Only run quickstart examples — rejected because it would miss regressions in advanced/benchmark scripts and conflict with SC-005. (b) Hard-code allow/deny lists inside the test — rejected for maintainability; manifest keeps decisions versioned and discoverable alongside documentation.
