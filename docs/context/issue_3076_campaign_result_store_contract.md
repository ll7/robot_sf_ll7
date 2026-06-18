# Issue #3076 Campaign Result Store Contract

Issue: [#3076](https://github.com/ll7/robot_sf_ll7/issues/3076)
Status: first contract slice; not benchmark evidence.

## Purpose

`scripts/tools/campaign_result_store.py` defines the compact canonical result-store surface for
research-engine campaign outputs. The store is intentionally reviewable: episode rows live in
`episodes.parquet`, while `summary.json`, `analysis.json`, `claim_card.yaml`,
`reproduction.md`, and empty `tables/` / `figures/` manifests keep the report and claim boundary
visible before derived outputs exist.

## Contract Boundary

This contract does not produce benchmark results, planner rankings, paper-facing claims, or durable
artifact publication by itself. It fails closed when required row-status or artifact-provenance
fields are missing, and it keeps derived tables/figures explicit so later report generators cannot
silently treat incomplete local `output/` rows as durable evidence.

## Required Episode Fields

Episode rows must include:

- `run_id`
- `episode_id`
- `planner`
- `scenario_id`
- `scenario_family`
- `seed`
- `row_status`
- `artifact_uri`
- `artifact_sha256`

Allowed `row_status` values are `native`, `adapter`, `diagnostic_only`, `fallback`, `degraded`,
`unavailable`, and `failed`.

## Validation

Focused proof:

```bash
uv run --extra analytics pytest tests/tools/test_campaign_result_store.py
uv run ruff check scripts/tools/campaign_result_store.py tests/tools/test_campaign_result_store.py
uv run ruff format --check scripts/tools/campaign_result_store.py tests/tools/test_campaign_result_store.py
git diff --check
```

## Follow-Up Boundary

This slice proves fixture round-trip through Parquet and DuckDB plus fail-closed provenance checks.
The next result-store work should connect real campaign outputs to this contract and then build
report generation on top of the validated store, not directly from ad hoc `output/` files.
