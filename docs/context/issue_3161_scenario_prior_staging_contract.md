# Issue #3161 dataset-backed scenario-prior staging contract (2026-06-27)

This note records the local, buildable half of issue #3161. The full comparison the issue asks for
-- do real-data scenario priors (Stanford Drone Dataset, SocNavBench ETH, AMV) change the scenario
space relative to the authored + trace-derived baseline from #2919 -- is **blocked on external
data**: no licensed dataset is staged in the repository, and Robot SF never ingests or
redistributes raw trajectories.

What ships here is a **metadata-only staging contract and checker**, so that when a dataset is later
staged the #2919 comparison can ingest a dataset-backed prior without guesswork.

> Evidence tier: `analysis_only` staging contract. This is **not** a staged dataset, an executed
> comparison, or a real-world realism claim. Every report stamps
> `evidence_boundary = staging_contract_only_no_dataset_ingest_no_comparison_run_no_realism_claim`.

## What was added

- Contract schema: `robot_sf/research/schemas/scenario_prior_staging_contract.v1.json`
- Checker module: `robot_sf/research/scenario_prior_staging_contract.py`
  (`load_scenario_prior_staging_contract`, `check_scenario_prior_staging_contract`).
- Example contract: `configs/research/scenario_prior_staging_contract_issue_3161.yaml`
  (SDD, SocNavBench ETH, AMV; all `blocked-external-input` today).
- CLI: `scripts/analysis/check_scenario_prior_staging_contract_issue_3161.py`.
- Tests: `tests/research/test_scenario_prior_staging_contract.py`.

## What the contract declares per dataset

- **Provenance / license** -- source URL, license, `license_status`
  (`license-gated` | `open` | `unknown`), and citation.
- **Distribution fields** -- the canonical scenario-prior parameter groups a dataset-backed prior
  would expose (`pedestrian_density`, `pedestrian_speed`, `timing_offset_s`, `spatial_offset_m`,
  `route_coordinate_m`, `clearance_distance_m`, ...). These are validated against the **live**
  `PARAMETER_GROUPS` vocabulary of the #2919 comparison harness
  (`scripts/analysis/compare_scenario_priors_issue_2919.py`), so the contract cannot silently drift
  from what the comparison can actually compute.
- **Explicit external-data blocker** -- a `blocked-external-input` dataset must name the staging
  issue(s) holding it (SDD #1497/#2657, SocNavBench ETH #1498, AMV #2000/#1585).
- **Redistribution** -- pinned to `none`; raw external data is never redistributed.

## Fail-closed behavior

- A declared `staged` dataset is reconciled against a live
  `manage_external_data.check_asset` presence probe (`--probe-live-staging`). If the files are not
  actually present, the dataset fails closed (`effective_staged = false`) so a comparison can never
  run on nothing.
- An unknown distribution field, a blocked dataset with no blocker issue, or a declared-vs-live
  mismatch makes the contract `invalid`.
- With no staged dataset (the state today), the contract resolves to `blocked-external-input` rather
  than substituting a synthetic stand-in -- matching the issue's acceptance / stop rule.

## Reproducibility commands

```bash
# Plan/check only -- no dataset ingest, no download:
uv run python scripts/analysis/check_scenario_prior_staging_contract_issue_3161.py

# Reconcile declared staging against live asset presence (no download):
uv run python scripts/analysis/check_scenario_prior_staging_contract_issue_3161.py \
    --probe-live-staging --require-ready

# Tests:
uv run python -m pytest tests/research/test_scenario_prior_staging_contract.py -q
uv run python -m pytest tests/ -k "scenario and prior" -q
```

## Status and next action

- Contract status today: `blocked-external-input` (no dataset staged). This is the expected,
  honest state.
- Next action: when SDD (#1497), SocNavBench ETH (#1498), or AMV (#2000) reaches `state:ready` and a
  dataset is staged via `scripts/tools/manage_external_data.py`, flip that dataset's
  `staging_status` to `staged`, run the checker with `--probe-live-staging`, and extend the #2919
  comparison harness to ingest the dataset-backed prior using the declared distribution fields.
- This PR does **not** run a benchmark campaign, submit any Slurm/GPU job, or edit paper/dissertation
  claims.

## Related

- Parent / sibling: #3057, #2919 (authored vs trace-derived comparison), #3192 (divergence metrics),
  #2918 (SDD/ETH/AMV extraction).
- Staging owners: `scripts/tools/manage_external_data.py` (`sdd`, `socnavbench-s3dis-eth`,
  `amv-calibration` AssetSpecs), `configs/data/sdd_staging_manifest.yaml`,
  `docs/context/issue_2657_sdd_staging.md`.
- Authored baseline registry: `configs/research/scenario_prior_cards_issue_2917.yaml`.
