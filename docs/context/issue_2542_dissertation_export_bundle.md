# Issue #2542 Dissertation Figure/Table Export Bundle (updated 2026-06-20)

Related issues: [#2542](https://github.com/ll7/robot_sf_ll7/issues/2542),
[#3203](https://github.com/ll7/robot_sf_ll7/issues/3203)

## Status

Current as of 2026-06-20 for payload presence and checksum integrity. The dissertation bundle
exporter is implemented in `scripts/tools/benchmark_publication_bundle.py dissertation-bundle`
with reusable manifest logic in `robot_sf/benchmark/artifact_publication.py`.

The current payloads come from the bounded Issue #3203 rerun of the original Issue #1023
scenario-horizon campaign config/seeds. That campaign is **not benchmark-success evidence**:
it exited `2` with `evidence_status=invalid`, preserved a PPO partial-failure row, and recorded
`snqi_contract_status=fail` under warn enforcement.

## Pilot Input

The current bundle uses compact table reports from the fresh local Issue #3203 campaign output:

- `campaign_table.md`
- `scenario_family_breakdown.md`

The source campaign command reconstructed the original Issue #1023 config and eval seeds from:

- `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml`
- `configs/scenarios/classic_interactions_francis2023.yaml`
- `configs/policy_search/scenario_horizons_h500.yaml`
- eval seeds `[111, 112, 113]`

The tracked payload files are durable compact evidence for dissertation packaging only. They do not
create a new benchmark, ranking, or Results-chapter claim.

## Pilot Command

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --output-root output/benchmarks/issue_3203 \
  --campaign-id issue3203_scenario_horizons_h500_reexport_2026-06-20 \
  --mode run \
  --log-level INFO
```

## Durable Proof

Small, reviewable proof copies are tracked under
`docs/context/evidence/issue_2542_dissertation_export_bundle/`:

- `artifact_spec.json`: selected source rows and claim-boundary metadata.
- `artifact_manifest.json`: generated manifest for the Issue #3203 table payloads.
- `checksums.sha256`: generated checksums for the tracked payload artifacts.
- `payload/artifacts/tab_issue_1023_campaign_table.md`: campaign table payload.
- `payload/artifacts/tab_issue_1023_scenario_family_breakdown.md`: scenario-family table payload.
- `docs/context/evidence/issue_3203_scenario_horizon_reexport_2026-06-20/reports/`: tracked
  compact source snapshots used as the manifest `source_root`.

The raw campaign output remains disposable under `output/benchmarks/issue_3203/...` and is not
tracked wholesale. Future dissertation-side registries should consume the manifest and compact
payloads, not the local output tree.

## Validation

- `scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/tools/test_benchmark_publication_bundle.py`
- `scripts/dev/run_worktree_shared_venv.sh -- uv run ruff check robot_sf/benchmark/artifact_publication.py scripts/tools/benchmark_publication_bundle.py tests/tools/test_benchmark_publication_bundle.py`
- `scripts/dev/run_worktree_shared_venv.sh -- uv run ruff format --check robot_sf/benchmark/artifact_publication.py scripts/tools/benchmark_publication_bundle.py tests/tools/test_benchmark_publication_bundle.py`
- `scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/benchmark_publication_bundle.py dissertation-bundle ...`
- `uv run python scripts/tools/benchmark_publication_bundle.py validate-dissertation-bundle --bundle-dir docs/context/evidence/issue_2542_dissertation_export_bundle --source-root docs/context/evidence/issue_3203_scenario_horizon_reexport_2026-06-20/reports --expected-source-command "uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml --output-root <disposable-output-root>/benchmarks/issue_3203 --campaign-id issue3203_scenario_horizons_h500_reexport_2026-06-20 --mode run --log-level INFO" --expected-source-commit 76d84347a40c669fb878b55489a2614917399bda`
- `uv run python scripts/tools/stale_artifact_detector.py docs/context/evidence/issue_2542_dissertation_export_bundle/artifact_manifest.json --json-out output/issue-3203/stale_artifact_report.json`
- `uv run python scripts/tools/reexport_readiness_preflight.py docs/context/evidence/issue_2542_dissertation_export_bundle/artifact_manifest.json --repo-root .` —
  read-only re-export readiness preflight (Issue #3203). Reports `fresh` / `stale` / `blocked` and
  the required re-export inputs (campaign config, generation script, source-commit provenance) so a
  stale bundle is never silently cited as current without checking that a bounded campaign can be
  reproduced here. It does not run the campaign or edit dissertation claims.
- `BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh`

## Boundary

This work creates a provenance and packaging workflow for selected figure/table artifacts. It does
not alter benchmark metrics, promote raw local `output/` trees as durable evidence, or claim that
the selected scenario-horizon tables are benchmark-success evidence. The Issue #3203 PPO
partial-failure and SNQI contract failure remain blocking caveats for any Results-style use.
