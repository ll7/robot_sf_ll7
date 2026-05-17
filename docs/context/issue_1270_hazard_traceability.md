# Issue #1270 Hazard Traceability Mapping

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1270>

## Summary

Issue #1270 adds `hazard_traceability.v1`, a compact mapping from scenario IDs or scenario families
to intended low-speed public-space hazards and supporting evidence fields.

## Implemented Surfaces

- `robot_sf/benchmark/schemas/hazard_traceability.v1.json` defines the JSON Schema.
- `robot_sf/benchmark/hazard_traceability.py` provides typed loading, JSON Schema validation,
  semantic validation, and coverage summary generation.
- `configs/benchmarks/hazard_traceability/low_speed_public_space_v1.yaml` is the first tracked
  traceability mapping.
- `docs/hazard_traceability.md` documents the contract boundary and authoring guidance.

## Boundary

Hazard traceability records intended coverage. It does not prove hazard mitigation, certify safety,
or replace benchmark execution. Any claim that cites hazard coverage still needs scenario
certification, benchmark artifacts, seeds, metrics, and outcome evidence.

## Validation

Targeted validation for this issue should include:

```bash
uv run pytest tests/benchmark/test_hazard_traceability.py -q
uv run ruff check robot_sf/benchmark/hazard_traceability.py tests/benchmark/test_hazard_traceability.py
uv run ruff format --check robot_sf/benchmark/hazard_traceability.py tests/benchmark/test_hazard_traceability.py
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```

Run the repository PR readiness gate after syncing with latest `origin/main` before PR handoff.
