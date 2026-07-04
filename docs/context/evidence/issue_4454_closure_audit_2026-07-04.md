# Issue #4454 Closure Audit

Plain-language summary: merged PR #4456 delivered the typed collision-event export requested by
issue #4454; this audit maps each acceptance criterion to merged evidence so the issue can be
closed by an authorized issue-closing follow-up.

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/4454>
- Merged implementation PR: <https://github.com/ll7/robot_sf_ll7/pull/4456>
- Merge time: 2026-07-04 17:47 UTC
- Audit date: 2026-07-04

## Claim Boundary

This is a closure-audit integration report only. It does not run a full benchmark campaign, submit
Slurm or GPU compute, regenerate release rows, change metric semantics, or edit paper or dissertation
claims. The delivered claim is additive instrumentation: episode rows can carry typed exact collision
event records, while existing scalar collision metrics remain backward-compatible termination and
aggregate indicators.

## Acceptance Mapping

| Acceptance criterion from #4454 | Delivered evidence | Status |
| --- | --- | --- |
| Episode rows contain additive typed collision-event records when event data is available. | PR #4456 bumps the canonical episode event ledger to `EpisodeEventLedger.v2`, adds `event_ledger.collision_events`, and emits the ledger from `run_map_episode` before validated JSONL writing. | Delivered by PR #4456. |
| Event records include partner type/id, collision time, relative speed at contact, clearance/source metadata, and exact event source. | `robot_sf/benchmark/event_ledger.py` defines `CollisionEventRecord` with `collision_partner_type`, `collision_partner_id`, `collision_time`, `relative_speed_at_contact`, `clearance_series_source`, and `exact_event_source`; validation rejects malformed typed events. | Delivered by PR #4456. |
| Pedestrian, static-geometry, boundary, and goal-artifact fixture tests cover classification behavior. | `tests/benchmark/test_event_ledger.py` parameterizes all four partner classes and checks normalized records preserve partner id, time, speed, clearance source, and exact source. `tests/benchmark/test_map_runner_utils.py` adds runtime helper coverage for pedestrian and obstacle/boundary classification paths. | Delivered by PR #4456. |
| Camera-ready retention preserves event data documented compact sidecar or native export. | PR #4456 emits typed events natively through the normal episode JSONL export path. The issue thread's latest maintainer comment says retained durable-artifact consumers were gate-hardened for bounded `SUPPORTED_EVENT_LEDGER_SCHEMA_VERSIONS = {v1, v2}` compatibility with v3 fail-closed fixtures, so camera-ready retention can carry typed events without a separate metric reinterpretation layer. | Delivered by PR #4456, via native episode export and durable-consumer schema gate hardening. |
| Existing scalar collision metrics and current claims are not reinterpreted. | PR #4456 keeps scalar fields backward-compatible and adds docs/changelog text that scalar `collision` / `outcome.collision_event` remains a termination indicator; safety claims should cite typed `event_ledger.collision_events`. | Delivered by PR #4456. |
| Documentation clarifies typed events are required for future safety-validity citation. | PR #4456 updates `docs/benchmark_spec.md` and `CHANGELOG.md` to document the typed event ledger and the scalar collision-flag boundary. | Delivered by PR #4456. |

## Closure Decision

All repository-visible acceptance criteria in the live issue thread are satisfied by merged PR #4456.
Issue #4454 is closable once an authorized actor can post the criterion-to-evidence closure comment
and close the issue. This audit did not post an issue comment because the queue authorization for this
run did not include `comment_issue_or_pr`.

## Local Verification

Audit-time validation for this docs-only slice:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/evidence/README.md --path docs/context/evidence/issue_4454_closure_audit_2026-07-04.md
git diff --check
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/benchmark/test_event_ledger.py tests/benchmark/test_map_runner_utils.py -q -k 'event_ledger or floors_exact_obstacle_collision_metrics'
```

The catalog edit was also parsed with PyYAML and checked for exactly one
`docs/context/evidence/issue_4454_closure_audit_2026-07-04.md` entry with `status: evidence` and
`freshness: evidence`.

No full benchmark campaign run, no Slurm or GPU job submitted, and no paper or dissertation claim
text changed.
