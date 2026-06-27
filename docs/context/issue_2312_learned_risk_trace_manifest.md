# Issue #2312 Durable Learned-Risk Trace Manifest

Date: 2026-06-27
Issue: <https://github.com/ll7/robot_sf_ll7/issues/2312>
Parent: <https://github.com/ll7/robot_sf_ll7/issues/1472>
Predecessor preflight: [`issue_2273_learned_risk_trace_preflight.md`](issue_2273_learned_risk_trace_preflight.md)

## Scope

This change adds the **durable trace manifest + baseline-artifact-URI contract** and a
**fail-closed validator** that #1472 learned-risk training can call before it runs. It does
**not** materialize traces, copy external data, run training, or submit SLURM. The trace bytes are
produced by the #1472 / #2441 SLURM runs; the manifest and validator are buildable now.

This closes the `Training trace manifest -> missing` gap recorded in the #2273 preflight: there is
now a tracked manifest and a mechanical readiness check, rather than only a launch packet plus a
contract fixture.

## What landed

- `robot_sf/training/learned_risk_trace_manifest.py` — `validate_trace_manifest()` returns a
  `training_readiness_decision` of `ready_for_training_handoff` or, fail-closed,
  `artifact_retrieval_blocked`. Malformed manifests raise `LearnedRiskTraceManifestError`.
- `scripts/validation/validate_learned_risk_trace_manifest.py` — CLI with decision-coded exit
  status: `0` ready, `2` structurally invalid, `3` blocked.
- `configs/training/learned_risk_trace_manifest_issue_2312.yaml` — the tracked manifest, currently
  in honest `blocked` state (pending baseline/trace aliases, `retrieval_status: blocked`).
- `tests/training/test_learned_risk_trace_manifest.py` — synthetic-manifest tests for the decision
  boundary plus the checked-in manifest.

Shared checksum logic was promoted to `sha256_file()` in
`robot_sf/training/learned_risk_launch_packet.py` and reused by both validators.

## Contract

A manifest (`schema_version: learned-risk-trace-manifest.v1`) is `ready_for_training_handoff` only
when **all** hold; any failure yields `artifact_retrieval_blocked`:

- `baseline_artifact_uri` and every `trace_artifacts` entry use a durable URI scheme and carry no
  placeholder alias (`:pending`, `tbd`, `todo`, ...);
- each concrete `trace_artifacts` URI has a recorded SHA-256 under `checksums`;
- `split_ids` cover the required `stress_slice` and `full_matrix` slices;
- `required_episode_fields` cover the learned-risk inputs (`scenario_id`, `seed`, `candidate_id`,
  `termination_reason`, `labels`);
- every required label (`collision`, `near_miss`, `low_progress`) in `label_availability` is
  `present`;
- `retrieval_status` is `available`.

Resolvability is the **local contract** (scheme, no placeholder, recorded digest, labels). A ready
decision means "locally contract-complete", not "bytes fetched and verified"; the durable digest is
what lets the training run verify the bytes it retrieves.

## Current decision

`artifact_retrieval_blocked`. The checked-in manifest still references `:pending` baseline and trace
aliases and has `label_availability: pending`, so training stays blocked — the truthful state until
the #1472 / #2441 SLURM run materializes and uploads versioned artifacts.

## Validation

```bash
uv run python scripts/validation/validate_learned_risk_trace_manifest.py \
  --config configs/training/learned_risk_trace_manifest_issue_2312.yaml --json   # exits 3 (blocked)
uv run python -m pytest tests/training/test_learned_risk_trace_manifest.py \
  tests/training/test_learned_risk_launch_packet.py -q
```

## Claim boundary

Data-preflight tooling only. It does not assert anything about the learned-risk model's quality; it
makes the trace/baseline readiness of #1472 mechanically checkable and fail-closed.
