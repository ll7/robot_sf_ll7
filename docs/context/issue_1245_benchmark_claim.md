# Issue #1245 Benchmark Claim Artifact 2026-05-18

## Goal

Issue #1245 adds a compact, schema-checked `benchmark_claim.v1` artifact that
separates paper-facing benchmark claims from training curves, validation runs,
local `output/` paths, and publication bundle packaging.

## Decision

`robot_sf_bench claim` is the public entrypoint. It builds a JSON artifact from:

- a frozen scenario matrix plus its expected SHA-256,
- policy metadata with explicit policy artifact SHA-256 values,
- distinct training, validation, and final benchmark episode JSONL groups,
- optional aggregate/statistical reports,
- repository/environment provenance including Git SHA, Python version, dependency
  group, and `uv.lock` SHA-256.

Final benchmark episodes are required. Training and validation episodes are
recorded only as distinct provenance groups, so they cannot silently become the
evidence for a paper-facing claim.

## Fail-Closed Boundary

Claim assembly fails before writing the artifact when:

- `scenario_matrix_sha256` is missing, malformed, or does not match the matrix,
- policy metadata lacks `schema_version` or any policy SHA-256,
- any episode JSONL input lacks a schema/version marker,
- required files are missing or not regular files.

This keeps the claim artifact as a reviewable boundary over already-produced
benchmark evidence rather than another benchmark execution engine.

## Validation

- `uv run pytest tests/benchmark/test_benchmark_claim.py -q`
  - First red state: collection failed because `robot_sf.benchmark.benchmark_claim`
    did not exist.
  - After implementation: `3 passed`.

## Related Surfaces

- Issue: https://github.com/ll7/robot_sf_ll7/issues/1245
- Schema: `robot_sf/benchmark/schemas/benchmark_claim.schema.v1.json`
- Builder/validator: `robot_sf/benchmark/benchmark_claim.py`
- CLI: `robot_sf/benchmark/cli.py` (`robot_sf_bench claim`)
- Fixture: `tests/fixtures/benchmark_claim/v1/`
- Release docs: `docs/benchmark_release_protocol.md`
