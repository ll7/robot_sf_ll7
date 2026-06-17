# Issue #3000: Map/Public-Smoke Result Provenance

## Related

- Issue: [#3000](https://github.com/ll7/robot_sf_ll7/issues/3000)
- Predecessor PR: [#2999](https://github.com/ll7/robot_sf_ll7/pull/2999) added the
  lightweight runner provenance contract that this note extends.
- Proof artifacts:
  - `tests/benchmark/test_map_runner_result_provenance.py`
  - `tests/validation/test_run_policy_search_candidate.py`

## Scope

Issue #3000 extends the result-artifact provenance contract from the lightweight
`robot_sf/benchmark/runner.py::run_batch` path to:

- `robot_sf/benchmark/map_runner.py::run_map_batch` summary payloads;
- `scripts/validation/run_policy_search_candidate.py` public smoke stage summaries.

Map-runner summaries now include a top-level `provenance` block with protocol version, commit hash,
run ID, Python version, invocation, config identity, seed identity, and artifact pointer status.
Policy-search stage summaries mirror that block as `result_provenance`, overwrite
`artifact_pointer_status` with the finalizer's fail-closed artifact classification, and append the
stage `jsonl_path`.

Map-runner `provenance` includes:

- `protocol_version`
- `commit_hash`
- `run_id`
- `python_version`
- `invocation`, shell-escaped with `shlex.join(sys.argv)`
- `artifact_pointer_status`
- `config_identity`: `schema_path`, `scenario_path`, `scenario_count`, `scenario_matrix_hash`,
  `algo`, `algo_config_path`, and `benchmark_profile`
- `seed_identity`: `suite_key`, `total_jobs`, and `written`

Policy-search combined override summaries use
`policy-search-combined-result-provenance.v1` and retain each per-run source under
`result_provenance.sources`.

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
