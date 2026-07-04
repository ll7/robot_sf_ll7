# Issue #4289 ATC Public External-Data Surface Terminality

Issue: <https://github.com/ll7/robot_sf_ll7/issues/4289>

Plain-language summary: the public Robot SF surface for the ATC pedestrian tracking dataset is
closed at registry, acquisition-documentation, and skip-if-absent loader shape-contract scope. No
private acquisition, raw data staging, benchmark run, or paper-facing claim is established here.

## Current Conclusion

Status: terminal for public external-data setup surface as of 2026-07-04.

The accepted public contract is:

- `scripts/tools/manage_external_data.py` owns canonical asset id `atc-pedestrian`, official ATR
  source URL, research-use/license notes, shared root subpath `atc_pedestrian`, trajectory CSV
  required-path group, terms/README required-path group, and fail-closed missing behavior.
- `docs/datasets/atc.md` records official acquisition, citation, chosen subset policy, expected
  layout, registry commands, skip-if-absent loader behavior, and the no-redistribution/no-benchmark
  claim boundary.
- `docs/external_data_setup.md` links ATC from the external-data setup index and summary table.
- `robot_sf/data/external/atc.py` implements the license-safe shape contract for locally staged
  daily CSV files only.
- `tests/tools/test_manage_external_data.py` and `tests/data/external/test_atc_shape.py` cover the
  registry metadata, missing-data status, synthetic fixture readiness, skip-if-absent behavior, and
  malformed CSV fail-closed paths without requiring official ATC bytes.

## Evidence

Merged slices:

- PR #4295: <https://github.com/ll7/robot_sf_ll7/pull/4295> documented ATC acquisition/layout.
- PR #4316: <https://github.com/ll7/robot_sf_ll7/pull/4316> added skip-if-absent shape-contract
  loader/tests.

Verification commands for this terminality pass:

```bash
uv run ruff check docs/datasets/atc.md docs/external_data_setup.md \
  docs/context/issue_4289_atc_public_surface_terminality.md
uv run pytest tests/tools/test_manage_external_data.py tests/data/external/test_atc_shape.py -q
uv run python scripts/tools/manage_external_data.py --json explain atc-pedestrian
uv run python scripts/tools/manage_external_data.py --json check atc-pedestrian
```

Expected absent-data behavior remains fail-closed: `check atc-pedestrian` should report
`ok: false` and `status: missing` when no local ATC copy is staged.

## Residual Boundary

Remaining work is intentionally outside #4289:

- private or manual acquisition of official ATC archives;
- raw data staging or durable artifact publication;
- benchmark consumers, campaign execution, planner comparison, or prediction-comparability
  evidence;
- dissertation, paper, or leaderboard claim edits.

If future work needs empirical ATC evidence, start from a new issue that names the consumer,
minimum valid staged subset, validation command, artifact provenance path, and claim boundary.
