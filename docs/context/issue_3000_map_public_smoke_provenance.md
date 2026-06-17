# Issue #3000: Map/Public-Smoke Result Provenance

## Scope

Issue #3000 extends the result-artifact provenance contract from the lightweight
`robot_sf/benchmark/runner.py::run_batch` path to:

- `robot_sf/benchmark/map_runner.py::run_map_batch` summary payloads;
- `scripts/validation/run_policy_search_candidate.py` public smoke stage summaries.

Map-runner summaries now include a top-level `provenance` block with protocol version, commit hash,
run ID, Python version, invocation, config identity, seed identity, and artifact pointer status.
Policy-search stage summaries mirror that block as `result_provenance` and overwrite
`artifact_pointer_status` with the finalizer's fail-closed artifact classification.

## Evidence Boundary

- Classification: `smoke/contract evidence`.
- Purpose: Makes map/public-smoke result artifacts self-describing enough for downstream closeout and
  handoff review.
- Paper-facing claims: This does not create benchmark-strength evidence or update research claims.

## Validation

Focused validation runs:

- `uv run pytest tests/benchmark/test_map_runner_result_provenance.py tests/validation/test_run_policy_search_candidate.py -q`
- `uv run ruff check robot_sf/benchmark/map_runner.py scripts/validation/run_policy_search_candidate.py tests/benchmark/test_map_runner_result_provenance.py tests/validation/test_run_policy_search_candidate.py`
- `uv run ruff format --check robot_sf/benchmark/map_runner.py scripts/validation/run_policy_search_candidate.py tests/benchmark/test_map_runner_result_provenance.py tests/validation/test_run_policy_search_candidate.py`
- `git diff --check`
