# Issue #1239 Human-Model Transfer Robustness

Issue: [#1239](https://github.com/ll7/robot_sf_ll7/issues/1239)
Evidence bundle:
[`docs/context/evidence/issue_1239_human_model_transfer_2026-05-18/`](evidence/issue_1239_human_model_transfer_2026-05-18/)

## Outcome

Issue #1239 now has a first, conservative smoke surface for human-model transfer robustness:

```text
configs/benchmarks/human_model_transfer_smoke_v1.yaml
```

The smoke config does not claim a full transfer benchmark. It makes the missing model axis explicit
by carrying `human_model_variant` and `human_model_source` through camera-ready campaign planner
specs, preflight matrix summaries, run metadata, campaign tables, and kinematics parity tables.

## Matrix Boundary

The first slice uses one native baseline and three Social-Navigation-PyEnvs adapter-backed proxy
rows:

* `native_social_force_default`: Robot SF native `social_force`, labeled
  `default_social_force` / `robot_sf_native`.
* `upstream_socialforce_proxy`: upstream Social-Navigation-PyEnvs SocialForce adapter.
* `upstream_sfm_helbing_proxy`: upstream Social-Navigation-PyEnvs SFM-Helbing adapter.
* `upstream_hsfm_new_guo_proxy`: upstream Social-Navigation-PyEnvs HSFM-New-Guo adapter.

These rows are a reproducibility and availability surface, not proof that the environment dynamics
have a first-class human-model switch. Adapter-backed rows must still report their execution mode,
availability status, and most likely failure reason. `not_available`, `failed`, `fallback`, or
`degraded` rows are limitations, not benchmark success.

## Canonical Commands

Preflight-only matrix proof:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/human_model_transfer_smoke_v1.yaml \
  --mode preflight \
  --campaign-id issue1239-human-model-transfer-preflight
```

Small run on a compatible host:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/human_model_transfer_smoke_v1.yaml \
  --campaign-id issue1239-human-model-transfer-smoke
```

## Validation

Test-first proof:

```bash
uv run pytest tests/benchmark/test_camera_ready_campaign.py::test_load_campaign_config_accepts_planner_group_and_paper_profile tests/benchmark/test_camera_ready_campaign.py::test_prepare_campaign_preflight_writes_matrix_summary -q
```

Initial result before implementation: failed because `PlannerSpec` had no
`human_model_variant` field and matrix summary rows did not include the new metadata.

Post-implementation result: `2 passed`.

Campaign regression tests:

```bash
uv run pytest tests/benchmark/test_camera_ready_campaign.py -q
```

Result: `71 passed`.

Lint:

```bash
uv run ruff check robot_sf/benchmark/camera_ready_campaign.py tests/benchmark/test_camera_ready_campaign.py
```

Result: all checks passed.

Docs proof consistency:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```

Result: passed for 10 changed files.

Post-`origin/main` PR-readiness gate:

```bash
PYTEST_NUM_WORKERS=8 BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

Result after committing the branch: Ruff passed, formatting reported `1251 files left unchanged`,
full pytest reported `3709 passed, 10 skipped`, the changed-file coverage check warned that
`robot_sf/benchmark/camera_ready_campaign.py` was at `90.1%`, and the TODO-docstring backlog
ratchet passed for 219 files and 1969 occurrences.

Preflight matrix proof:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/human_model_transfer_smoke_v1.yaml \
  --mode preflight \
  --campaign-id issue1239-human-model-transfer-preflight
```

Result: four matrix rows with `scenario_count: 1`, `resolved_seeds: [111]`, and explicit
`human_model_variant` / `human_model_source` values.

Smoke run:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/human_model_transfer_smoke_v1.yaml \
  --campaign-id issue1239-human-model-transfer-smoke \
  --skip-publication-bundle
```

Result: exit code `2`, because the campaign correctly did not meet benchmark-success criteria.
`native_social_force_default` produced one episode with `availability_status: available`. The three
Social-Navigation-PyEnvs proxy rows failed closed with `availability_status: failed` because
`output/repos/Social-Navigation-PyEnvs` was not present.

Compact evidence was promoted to
[`docs/context/evidence/issue_1239_human_model_transfer_2026-05-18/`](evidence/issue_1239_human_model_transfer_2026-05-18/).

## Follow-Up Boundary

A full #1239 research result still needs a larger scenario subset, paired degradation reporting, and
clear interpretation of whether adapter differences are true human-model transfer or only
planner-wrapper variation. This first slice exists to prevent future runs from hiding that axis in
planner names alone.
