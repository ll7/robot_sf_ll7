# Issue #1126 SDD Curation Readiness Preflight

Status: Current.
Scope: curation-step readiness for first real Stanford Drone Dataset (SDD) derived scenario set.
This note does not record a benchmark campaign or paper-facing claim.

## Current State

PR #3765 added fail-closed SDD curation preflight. PR #4564 added a metadata-only decision packet.
PR #4616 fixed the packet's generated importer command so it uses the importer's real
`--annotations`, `--out-dir`, and `--meters-per-pixel` arguments.

PR #4638 pinned the reviewed bring-your-own SDD annotation tree in
`configs/data/sdd_staging_manifest.yaml`:

- `expected_tree_sha256`: `66dec2c82b0a01b23bf9fa9acef352af86549e7ea749811ea4ef9c47003d4acf`
- matched files: 60 `annotations.txt` files
- total size: 444959624 bytes

The pin lets `scripts/tools/manage_external_data.py sdd-mode` report `dataset_backed_prior` when
the exact local tree is available under an approved external-data root. It does not commit raw SDD
files and does not commit local license acknowledgment.

The first real import/smoke slice is recorded in
[`evidence/issue_1126_real_sdd_smoke_2026-07-06.md`](evidence/issue_1126_real_sdd_smoke_2026-07-06.md).
It selects `annotations/bookstore/video0/annotations.txt`, imports a four-pedestrian generated
scenario into ignored `output/`, and runs two CPU `simple_policy` smoke jobs. The candidate is
`exploratory_only` because both smoke horizons timed out; that timeout evidence does not close the
benchmark-ready scenario-set criterion. The final integration audit is
[`evidence/issue_1126_final_integration_2026-07-06.md`](evidence/issue_1126_final_integration_2026-07-06.md).

## Owners

- SDD staging gate: `scripts/tools/manage_external_data.py`
  (`load_sdd_staging_spec`, `validate_sdd_staging`, `resolve_sdd_scenario_prior_mode`) and
  `configs/data/sdd_staging_manifest.yaml`.
- SDD importer: `scripts/tools/import_sdd_scenarios.py`.
- Curation readiness gate: `scripts/tools/sdd_curation_preflight.py`.

## Validation

With `ROBOT_SF_EXTERNAL_DATA_ROOT` pointing at the reviewed local BYO data tree:

```bash
scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/manage_external_data.py --json sdd-mode
# dataset_backed_prior, dataset_backed true

scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/sdd_curation_preflight.py --json
# dataset_backed_prior, benchmark_promotion_allowed false until selected annotation probe passes
```

Without the pinned tree present, the same commands still fail closed as `proxy_schema_smoke` and
must not be treated as benchmark evidence.

## Closure Boundary

First real SDD curation evidence is complete only at `exploratory_only` strength. #1126 remains open
until a tuned/calibrated selected candidate or alternate SDD scene reaches benchmark-ready smoke
criteria. Out of scope here: raw SDD commits, benchmark campaign runs, Slurm/GPU submission, and
paper/dissertation claim edits.
