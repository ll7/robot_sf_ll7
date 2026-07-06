# Issue #1470 Closure Audit

Date: 2026-07-06
Issue: <https://github.com/ll7/robot_sf_ll7/issues/1470>
Successor (execution): <https://github.com/ll7/robot_sf_ll7/issues/2441>
Launch note: [`issue_1397_oracle_imitation_launch_packet.md`](../issue_1397_oracle_imitation_launch_packet.md)
Trace evidence bundle: [`evidence/issue_1470_oracle_imitation_traces_12911_2026-06-17/`](issue_1470_oracle_imitation_traces_12911_2026-06-17/)

## Purpose

Closure audit for #1470 (`data: collect oracle imitation dataset from launch packet`). This note
maps each acceptance criterion to merged-PR / completed-Slurm evidence and to a locally reproduced
validation run, then records the closure decision. It asserts only data/trace-collection readiness;
it does not materialize the final imitation NPZ dataset, run training, submit Slurm/GPU, or promote
any model/benchmark/paper claim.

## Authoritative scope

The issue body's `agent-exec-spec:v1` block (appended 2026-06-20) is the authoritative scoping and
overrides the older issue-body wording. It states plainly: *"The dataset collection RUN is SLURM
(`resource:slurm`); the agent-executable slice is the launch-packet preflight + manifest contract
validation."* It also fixes the boundary: *"This issue stops after durable dataset manifest +
checksums + pointers exist; imitation training is the separate #1496."* The 2026-06-16 maintainer
triage reinforces this — treat #1470 as the **lane controller**, move execution to #2441 and
downstream artifact governance to #2655, with closeout states `dataset_ready`,
`artifact_retrieval_blocked`, or `failed_closed`.

Per the repository COMPLETE-FIRST rule, an issue whose only remaining work is a compute/Slurm run
counts as complete for the agent-executable contract. Here the Slurm run additionally **already
completed** (#2441, job `12911`) and its traces + manifest + checksums are tracked on `main`
(#2989), so the closeout state is `dataset_ready`.

## Acceptance criteria → evidence

| Acceptance criterion (issue "Accept" list) | Status | Evidence |
| --- | --- | --- |
| Local launch-packet validation passes at the collection commit | Met | PR #1469 (`fe72b911d`, merged 2026-05-24) added the packet + validator; reproduced today on `origin/main` `405eb5b5a`: `status=valid`, `dataset_id=issue_1397_oracle_imitation_v1`, 6 scenarios, 12 episode ids (below) |
| Dataset collection completes for the declared split packet, or fails closed with row-level reasons | Met | #2441 (closed) ran the Slurm collection across train/validation/evaluation splits: jobs `12669`, `12762`/`12763`, `12764`/`12765`, and re-run `12911` all `COMPLETED` with `exit 0`; per-episode termination/failure reasons recorded in the manifest (`success=4`, `collision=1`, `max_steps=1` for the tracked `12911` train split) |
| Manifest records source candidate, config, scenario ids, seeds by split, episode ids, generating commit, artifact paths, and checksums | Met | Tracked `oracle_candidate_trace_manifest.json` in the #2989 bundle records `candidate=hybrid_rule_v3_static_margin0_waypoint2`, `candidate_config_path`, scenario families (`classic`/`francis2023`/`nominal`), `git_hash=dcb14927…`, per-group artifact paths, and `dataset_boundary`; `SHA256SUMS` + `source_slurm_checksum_manifest.sha256` carry the checksums |
| Durable artifact pointers exist for the (trace) dataset and manifest | Met | PR #2989 (`5b9497045`, merged 2026-06-17) tracks the six-row JSONL, source manifest, source Slurm checksum manifest, and bundle `SHA256SUMS` under `docs/context/evidence/issue_1470_oracle_imitation_traces_12911_2026-06-17/` — a durable git-tracked pointer, retrievable independent of worktree-local `output/` |
| No train/validation/evaluation leakage is detected | Met | Validator confirms disjoint split seeds (train `201–206`, validation `101–103`, evaluation `111–113`); train disjoint from `paper_eval_s20`. Reproduced today (`status=valid`) |

## Reproduced validation (2026-07-06, `origin/main` @ `405eb5b5a`)

```bash
uv run python scripts/validation/validate_oracle_imitation_launch_packet.py \
  --config configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml --json
# status=valid, dataset_id=issue_1397_oracle_imitation_v1, scenario_count=6, episode_count=12
# (training_ready=false — intentional; see residual below)

uv run python -m pytest tests/ -k "imitation and (manifest or packet)" -q
# 54 passed, 11695 deselected
```

The `training_ready=false` flag and the `:pending` W&B aliases in the packet's `collection_roots` /
`artifact_paths` are the **intended** signal that the downstream durable-URI registry is not yet
resolved. The regression tests
`test_validate_launch_packet_rejects_pending_trace_uri_registry` and
`test_validate_launch_packet_accepts_training_ready_trace_uri_registry` pin that gate, which is owned
by #2655 (registry) and consumed by #1496 (training) — not by #1470's trace-collection scope.

## Closure decision

**Close #1470.** Every acceptance criterion in the issue's scope — launch-packet validity, completed
split collection, a checksummed manifest, durable git-tracked trace pointers, and no split leakage —
is met by merged/closed work (#1469, #2441, #2989) and reproducibly re-validated on `origin/main`.
This is a *stronger* closeout than a typical agent-executable-only audit: the Slurm collection run
itself completed and produced durable tracked evidence.

### Residual (owned by separate open issues — out of #1470 scope)

- **#2655** (`training: add durable trace URI registry for oracle imitation artifacts`) — resolve
  the `:pending` W&B aliases into a durable trace-URI registry and flip `training_ready` to `true`.
  The launch-packet validator already fails closed on the pending state; this is downstream artifact
  governance, explicitly moved off #1470 in the 2026-06-16 triage.
- **#1496** (`research: benchmark oracle imitation warm-start after durable dataset collection`) —
  the imitation-training/benchmark step, which the issue body scopes out ("imitation training is the
  separate #1496"). `test_real_issue_1496_manifest_is_blocked_on_dataset` confirms #1496 stays
  fail-closed until the durable dataset exists.

Neither residual is a #1470 acceptance criterion, so closing #1470 does not lose the tracking trail.
