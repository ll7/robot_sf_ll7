# Issue #6875 — Camera-Ready Report Crosswalk Producer

## Status and ownership

Issue [#6875](https://github.com/ll7/robot_sf_ll7/issues/6875) wires the
diagnostic-only `report_crosswalk.v1` contract from
`robot_sf/benchmark/report_crosswalk.py` into the canonical camera-ready
campaign finalization path:

- producer: `robot_sf/benchmark/camera_ready/_crosswalk_producer.py`
- owner seam: `_finalize_campaign_outputs()` in
  `robot_sf/benchmark/camera_ready/campaign.py`
- sidecar: `reports/report_crosswalk.v1.json`
- focused tests: `tests/benchmark/test_camera_ready_crosswalk_producer.py`

The sidecar is written before publication-bundle export and its repo-relative
pointer is added to the final `campaign_summary.json` artifact map. Existing
campaign metrics, planner rows, ranking fields, and release-gate fields are
not rewritten by the producer.

## Input and output contract

The producer consumes the run-entry `episodes_path` values emitted by the
camera-ready campaign. Each JSONL row is mapped using the canonical
`outcome.route_complete`, `outcome.collision_event`, and
`metrics.comfort_exposure` fields. Older rows with an explicit
`metrics.comfort` value remain readable. Optional diagnosis payloads are read
only from `diagnosis_payload` or `failure_diagnosis` and are validated by the
upstream crosswalk.

Every source artifact carries its resolved path, SHA-256, run identity, and
record counts. Every episode carries episode/scenario/seed/planner identity,
the source-artifact receipt, and the retained `result_provenance` block. Missing
or incomplete identity is marked `incomplete`; malformed JSONL lines are
listed in `provenance.invalid_source_records` and do not enter the episode
denominator.

The producer does not deserialize arbitrary JSON into an
`ExecutionDeviationResult`. If a serialized execution-deviation field is
present without a validated result object, the sidecar marks that input
`invalid` with an explicit reason and keeps execution-deviation availability at
zero. Missing inputs remain `unavailable`; upstream `fallback` and `degraded`
diagnosis states remain explicit and are not promoted to benchmark evidence.

## Validation and claim boundary

Focused proof:

```text
scripts/dev/run_worktree_shared_venv.sh -- python -m pytest \
  tests/benchmark/test_camera_ready_crosswalk_producer.py \
  tests/benchmark/test_report_crosswalk.py \
  tests/benchmark/test_camera_ready_campaign.py -q
scripts/dev/run_worktree_shared_venv.sh -- ruff check \
  robot_sf/benchmark/camera_ready/_crosswalk_producer.py \
  robot_sf/benchmark/camera_ready/campaign.py \
  tests/benchmark/test_camera_ready_crosswalk_producer.py
scripts/dev/run_worktree_shared_venv.sh -- ruff format --check \
  robot_sf/benchmark/camera_ready/_crosswalk_producer.py \
  robot_sf/benchmark/camera_ready/campaign.py \
  tests/benchmark/test_camera_ready_crosswalk_producer.py
```

This is an artifact-contract and diagnostic-reporting change. It does not
establish causality, safety, intervention effectiveness, planner ranking,
generalization, or benchmark/paper validity, and it does not run or authorize
a campaign.

## Related context

- [Issue #6871 report crosswalk](issue_6871_report_crosswalk.md)
- [Issue #6872 cross-context validity matrix](issue_6872_cross_context_validity_matrix.md)
