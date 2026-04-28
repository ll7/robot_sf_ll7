# Benchmark Release 0.0.2 Execution Log (2026-04-13)

## Goal

Run a full, paper-facing benchmark release with all currently configured planners, keep execution
alive across remote disconnects (tmux), and produce release-ready artifacts for downstream paper
selection decisions.

## Scope

* Include all planners present in the all-planners camera-ready contract:
`prediction_planner` , `goal` , `social_force` , `orca` , `ppo` , `socnav_sampling` , `sacadrl` , 
`socnav_bench` .
* Use benchmark release protocol entrypoint (`run_benchmark_release.py`) instead of ad-hoc campaign
  invocation to preserve release-provenance stamping and publication bundle behavior.
* Target release tag: `0.0.2`.

## Canonical Inputs Used

* New campaign config:
 `configs/benchmarks/paper_experiment_matrix_all_planners_v1.yaml`

* New release manifest:
 `configs/benchmarks/releases/paper_experiment_matrix_all_planners_v1_release_v0_0_2.yaml`

* Comparability map updated for all planners:
`configs/benchmarks/alyassi_comparability_map_v1.yaml` (added `socnav_bench` mapping)

## Reasoning And Decisions

1. Release wrapper over raw camera-ready runner
   - Decision: use `scripts/tools/run_benchmark_release.py` .
   - Why: this is the documented full benchmark release protocol and guarantees release provenance

     merge + required artifact validation + publication bundle export for valid runs.

2. Paper-facing all-planners config instead of non-paper config
   - Decision: create `paper_experiment_matrix_all_planners_v1.yaml` rather than using

`camera_ready_all_planners.yaml` directly.

   - Why: release protocol requires `paper_facing=true` contract checks and explicit

`planner_group` declarations.

3. Single-worker execution for release reproducibility
   - Decision: set `workers: 1` in the release campaign config.
   - Why: aligns with reproducibility guidance to reduce ordering variance in frozen releases.

4. SocNav prerequisite policy handling
   - Decision: keep planner-specific policies aligned with existing all-planners conventions

     ( `fail-fast` for `socnav_sampling` and `socnav_bench` ; fallback for `orca` / `sacadrl` ).

   - Tradeoff: stricter policies may fail faster if third-party assets are missing, but this

     preserves fail-closed benchmark semantics.

## Assumptions

* Model and planner dependencies required by enabled planners are available in this environment or
  will fail clearly under benchmark fail-closed policy.
* GitHub release publication (asset upload) can be executed after campaign completion if `gh`
  authentication is present.
* Tag naming `0.0.2` is accepted as the intended benchmark release tag, independent of Python
  package version.

## Commands And Evidence

Preflight validation:

```bash
source .venv/bin/activate
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_all_planners_v1_release_v0_0_2.yaml \
  --mode preflight \
  --label rel_0_0_2_preflight
```

Result: `manifest_validation.status = valid` .

Long-running release execution in tmux:

```bash
tmux new-session -d -s bench_release_0_0_2 \
  'cd /home/luttkule/git/robot_sf_ll7 && source .venv/bin/activate && \
   uv run python scripts/tools/run_benchmark_release.py \
     --manifest configs/benchmarks/releases/paper_experiment_matrix_all_planners_v1_release_v0_0_2.yaml \
     --label rel_0_0_2_full 2>&1 | tee output/benchmarks/camera_ready/release_0_0_2_tmux.log'
```

Active campaign root detected:

* `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_rel_0_0_2_full_20260413_153749`

Live log path:

* `output/benchmarks/camera_ready/release_0_0_2_tmux.log`

## Current Status

* Historical run (`2026-04-13`) reached `benchmark_failed` with 7/8 planners successful.
* Root cause identified from campaign summary: `socnav_bench` failed fail-fast preflight because
  optional dependency `skfmm` was missing.
* Dependency remediation applied on `2026-04-14` via `uv sync --all-extras`.
* Verification check passed: `python -c "import skfmm"`.
* Fresh detached tmux rerun launched:
  + session: `bench_release_0_0_2_rerun`
  + log: `output/benchmarks/camera_ready/release_0_0_2_rerun_tmux.log`
  + label: `rel_0_0_2_full_rerun`
* GitHub release object now exists for tag `0.0.2`:
  + `https://github.com/ll7/robot_sf_ll7/releases/tag/0.0.2`
* Rerun root:
  + `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_rel_0_0_2_full_rerun_20260414_081244`
* Rerun again reached `benchmark_failed`.
* Updated root cause: Python prerequisites are present, but licensed SocNavBench data assets are not
  present on this machine.
* Asset validation evidence:
  + command: `/home/luttkule/git/robot_sf_ll7/.venv/bin/python scripts/tools/prepare_socnav_assets.py --report-json output/tmp/socnav_asset_report.json`
  + report: `output/tmp/socnav_asset_report.json`
  + missing required directories: `wayptnav_data`,  `sd3dis/stanford_building_parser_dataset`, 
 `sd3dis/stanford_building_parser_dataset/traversibles`

* Local machine search found a SocNavBench source clone at `output/repos/SocNavBench`, but it only
  provides code plus `wayptnav_data` ; the required SD3DIS dataset root and traversibles are not
  present anywhere under `/home/luttkule` .
* Post-run analysis artifacts were generated for the failed rerun:
  + `reports/campaign_analysis.json`
  + `reports/campaign_analysis.md`
  + `reports/scenario_difficulty_analysis.json`
  + `reports/scenario_difficulty_analysis.md`
* Publication bundle export and release asset upload remain blocked until the required SocNavBench
  assets are staged locally and the full 8-planner run completes benchmark-valid.

## Open Follow-Ups

1. Stage required SocNavBench assets using the license-safe runbook in
`docs/socnav_assets_setup.md` or an existing local `output/SocNavBench` clone.
2. Validate staged assets:
  + `/home/luttkule/git/robot_sf_ll7/.venv/bin/python scripts/tools/prepare_socnav_assets.py --report-json output/tmp/socnav_asset_report.json`
3. Re-run the release manifest after assets are present and verify:
  + `release/release_result.json`
  + `reports/campaign_summary.json`
  + publication bundle paths in summary
4. Execute camera-ready release publication upload for tag `0.0.2`:
   - `scripts/tools/publish_camera_ready_release.py --execute-upload`

5. Record final release URLs and any planner `not_available`/`failed` caveats.
