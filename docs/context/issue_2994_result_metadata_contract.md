# Issue #2994: Episode Result Provenance Metadata Contract

## Scope

Issue #2994 adds optional per-episode provenance metadata on the canonical benchmark runner path (`robot_sf/benchmark/runner.py`).

The provenance block captures the following execution information per episode:
- `protocol_version`: retrieved from `robot_sf.benchmark.release_protocol.BENCHMARK_PROTOCOL_VERSION` (nominally `0.1.0`).
- `commit_hash`: the Git commit reported by the benchmark runner at artifact generation time.
- `base_seed`: the base seed supplied for the batch run.
- `run_id`: a stable, per-run UUID generated once per `run_batch` execution.
- `python_version`: the running Python version retrieved via `platform.python_version()`.
- `invocation`: the command-line arguments used for executing the benchmark run (if available via `sys.argv`).
- `config_identity`: a compact identity block with the schema path, algorithm name, optional
  algorithm config path, scenario count, and scenario matrix hash.

This metadata is embedded at the top level of each episode dictionary using a schema-compatible `"provenance"` key.

## Evidence Boundary

- Classification: `smoke/contract evidence`.
- Purpose: Verifies that metadata tracking and run-level identifiers can be safely injected on the runner path without regression on existing metrics and while maintaining full schema compatibility.
- Paper-Facing Claims: This does **not** constitute paper-grade benchmark evidence, metric validation, or dissertation-grade claim updates.

## Validation

Focused validation runs:
- `uv run pytest tests/test_runner_smoke.py tests/contract/test_episode_schema.py -q`
- `uv run ruff check robot_sf/benchmark/runner.py tests/test_runner_smoke.py`
- `uv run ruff format --check robot_sf/benchmark/runner.py tests/test_runner_smoke.py`
- `git diff --check`
