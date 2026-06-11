# Issue #2542 Dissertation Figure/Table Export Bundle (2026-06-11)

Related issue: [#2542](https://github.com/ll7/robot_sf_ll7/issues/2542)

## Status

Current as of 2026-06-11. The dissertation bundle exporter is implemented in
`scripts/tools/benchmark_publication_bundle.py dissertation-bundle` with reusable manifest logic in
`robot_sf/benchmark/artifact_publication.py`.

## Pilot Input

The pilot uses tracked compact table evidence from
`docs/context/evidence/issue_1023_scenario_horizons_local_full_2026-05-06/reports/`:

- `campaign_table.md`
- `scenario_family_breakdown.md`

These files are durable repository evidence, but they remain historical Issue #1023 evidence. The
pilot proves the export/provenance workflow only; it does not create a new benchmark or dissertation
claim.

## Pilot Command

```bash
uv run python scripts/tools/benchmark_publication_bundle.py dissertation-bundle \
  --source-root docs/context/evidence/issue_1023_scenario_horizons_local_full_2026-05-06/reports \
  --out-dir output/dissertation_export \
  --bundle-name issue_2542_scenario_tables \
  --artifact-spec docs/context/evidence/issue_2542_dissertation_export_bundle/artifact_spec.json \
  --command "uv run python scripts/tools/benchmark_publication_bundle.py dissertation-bundle --source-root docs/context/evidence/issue_1023_scenario_horizons_local_full_2026-05-06/reports --out-dir output/dissertation_export --bundle-name issue_2542_scenario_tables --artifact-spec docs/context/evidence/issue_2542_dissertation_export_bundle/artifact_spec.json --commit <commit>" \
  --commit <commit> \
  --overwrite
```

## Durable Proof

Small, reviewable proof copies are tracked under
`docs/context/evidence/issue_2542_dissertation_export_bundle/`:

- `artifact_spec.json`: selected source rows and claim-boundary metadata.
- `artifact_manifest.json`: generated manifest copied from the pilot bundle.
- `checksums.sha256`: generated checksums for the copied payload artifacts.

The payload files remain under disposable `output/dissertation_export/...` during the smoke run and
are not committed. Future dissertation-side registries should consume the manifest shape, not local
worktree output paths.

## Validation

- `scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/tools/test_benchmark_publication_bundle.py`
- `scripts/dev/run_worktree_shared_venv.sh -- uv run ruff check robot_sf/benchmark/artifact_publication.py scripts/tools/benchmark_publication_bundle.py tests/tools/test_benchmark_publication_bundle.py`
- `scripts/dev/run_worktree_shared_venv.sh -- uv run ruff format --check robot_sf/benchmark/artifact_publication.py scripts/tools/benchmark_publication_bundle.py tests/tools/test_benchmark_publication_bundle.py`
- `scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/benchmark_publication_bundle.py dissertation-bundle ...`
- `BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh`

## Boundary

This work creates a provenance and packaging workflow for selected figure/table artifacts. It does
not alter benchmark metrics, promote local `output/` files as durable evidence, or claim that the
selected Issue #1023 tables are dissertation-ready without downstream review.
