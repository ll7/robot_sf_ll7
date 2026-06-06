# Issue #2496 Multimodal Prediction Contract Smoke (2026-06-06)

Status: executable contract smoke, not benchmark evidence.

Related surfaces:

- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2496
- Parent benchmark contract: https://github.com/ll7/robot_sf_ll7/issues/2476
- Interface prerequisite: PR #2495 / issue #2475, merged probabilistic prediction interface
- Smoke config: `configs/benchmarks/multimodal_prediction_smoke_issue_2496.yaml`
- Smoke runner: `scripts/validation/run_multimodal_prediction_contract_smoke.py`
- Parent proposal artifact: `configs/benchmarks/multimodal_prediction_contract_issue_2476.yaml`

## Result

The smoke runner materializes the first executable trace contract for multimodal pedestrian
prediction rows. It emits native rows for:

- `reactive_no_prediction`,
- `single_trajectory_prediction`,
- `multimodal_equal_weight`,
- `multimodal_confidence_weighted`.

It also emits expected fail-closed rows for missing and degraded prediction inputs. These rows must
use `not_available` or `failed` status and must preserve a non-empty
`fallback_or_degraded_reason`.

The rows include `prediction_sample_count` as an additional smoke-only field so the runner can
check the parent contract invariant that sample count matches per-pedestrian hypothesis count.

## Claim Boundary

This is contract evidence only. The runner uses deterministic fixtures and does not execute a
planner campaign, measure prediction quality, or compare planning performance. It should be used to
guard trace shape, native/fail-closed status semantics, and row discoverability before a broader
benchmark is proposed.

## Validation Path

Canonical command:

```bash
uv run python scripts/validation/run_multimodal_prediction_contract_smoke.py
```

Targeted pytest:

```bash
uv run pytest tests/validation/test_multimodal_prediction_contract_smoke.py -q
```

The default runner writes disposable local artifacts under
`output/benchmarks/multimodal_prediction_smoke_issue_2496/`:

- `rows.jsonl`,
- `summary.json`,
- `summary.md`.

The tracked proof surface is the config, runner, and test coverage. Local `output/` artifacts are
not durable benchmark evidence.

## Follow-Up Boundary

The next benchmark-strengthening step should replace the deterministic fixture rows with a true
runner path that executes candidate planners on a smoke-sized scenario matrix while preserving the
same required trace fields and fail-closed status behavior.
