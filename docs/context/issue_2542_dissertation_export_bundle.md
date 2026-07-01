# Issue #2542 Dissertation Figure/Table Export Bundle

Related issues: [#2542](https://github.com/ll7/robot_sf_ll7/issues/2542),
[#3203](https://github.com/ll7/robot_sf_ll7/issues/3203),
[#3266](https://github.com/ll7/robot_sf_ll7/issues/3266)

## Status

Current as of 2026-07-01: the dissertation export bundle remains a payload-complete
discussion/provenance artifact, not paper-facing Results evidence.

The tracked bundle under `docs/context/evidence/issue_2542_dissertation_export_bundle/` still uses
the historical 2026-06-20 #3203 source snapshots. Those payloads are retained because they prove the
bundle mechanics, manifest, checksums, and selected table files, but they are not benchmark-success
evidence.

Issue #3203 was rerun on 2026-07-01 after the #3266 PPO/SNQI smoke repair. That fresh bounded
campaign fixed the stale PPO blocker:

- campaign exit code: `0`
- campaign status: `benchmark_success`
- evidence status: `valid`
- PPO row status: `ok`
- PPO execution mode: `native`
- PPO learned-policy contract: `pass`
- unexpected failed rows: `0`
- fallback/degraded rows counted as success: `0`

The fresh rerun was not promoted into this dissertation bundle because the SNQI contract still
failed. The readiness checker classifies the 2026-07-01 packet as `diagnostic_only` because
`snqi_contract_status=fail`, with rank-alignment Spearman
`-0.19999999999999998` below the `0.3` fail threshold.

## Durable Proof

- Historical bundle payloads:
  `docs/context/evidence/issue_2542_dissertation_export_bundle/`
- Historical #3203 source snapshots:
  `docs/context/evidence/issue_3203_scenario_horizon_reexport_2026-06-20/reports/`
- Fresh diagnostic #3203 rerun packet:
  `docs/context/evidence/issue_3203_scenario_horizon_reexport_2026-07-01/`

Raw campaign output remains disposable under `output/benchmarks/issue_3203/...` and is not tracked
wholesale.

## Boundary

Do not cite the scenario-horizon table bundle as benchmark-success, ranking, or Results-chapter
evidence. It is suitable only for discussion/provenance wording unless a future bounded rerun either
passes the SNQI contract or predeclares a narrower claim boundary that explicitly scopes SNQI out.

## Validation

- `uv run python scripts/validation/check_scenario_horizon_results_readiness.py docs/context/evidence/issue_3203_scenario_horizon_reexport_2026-07-01/reports/campaign_summary.json --json`
- `uv run pytest tests/benchmark/test_scenario_horizon_readiness_issue_3203.py`
- `uv run pytest tests/docs/test_dissertation_evidence_ledger.py tests/docs/test_dissertation_gap_report.py`
