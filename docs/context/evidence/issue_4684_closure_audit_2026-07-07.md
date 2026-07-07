# Issue #4684 Closure Audit

This audit maps the campaign fail-fast acceptance criteria to merged repository evidence and
separates the remaining private-operations deployment proof from the in-repository contract.

- Issue: [#4684](https://github.com/ll7/robot_sf_ll7/issues/4684)
- Audit date: 2026-07-07
- Evidence status: repository-side contract verified; private-operations deployment proof remains.
- Claim boundary: code and CPU (central processing unit) tests only. This is not a full benchmark
  campaign, Simple Linux Utility Resource Management (SLURM) submission, graphics processing unit
  run, or paper-facing result claim.
- Fragmentation guard: two issue #4684 PRs merged within 24 hours, so this report is an
  integration slice instead of another guardrail-only packet.

## Acceptance Evidence

| Acceptance criterion | Repository evidence | Status |
| --- | --- | --- |
| L0 blocks submission on any unresolvable arm, including unknown algorithm and missing checkpoint, through the real campaign factory path. | PR [#4708](https://github.com/ll7/robot_sf_ll7/pull/4708) merged `scripts/benchmark/run_issue_4205_codesign_loop_campaign.py` changes that run full arm-resolution preflight and fail closed before campaign outputs. Its tests in `tests/benchmark/test_issue_4367_codesign_campaign_runner.py` cover unknown algorithm, missing hydrated arm, and missing checkpoint cases. | Met in repo. |
| L1 phase-0 arm smoke runs before the real matrix and surfaces broken arms early. | PR #4708 added mandatory phase-0 smoke in full mode. The benchmark runner tests assert phase-0 smoke metadata, one-scenario smoke calls, and ordering before the full matrix. | Met in repo. |
| L2 risky-first arm ordering runs historically risky arms before healthier arms. | PR #4708 added `RISKY_FIRST_ARM_KEYS` and asserts the smoke/full execution order starts with `ppo_frozen_cbf_on`, then `ppo_frozen_wrapper_on`, then `ppo_frozen`. | Met in repo. |
| L3 live per-arm status stream is written incrementally for operations polling. | PR #4708 writes `live_arm_status.json` and `arm_status.jsonl`; tests assert running/completed events and the live status payload. | Met in repo. |
| L4 operations-side polling can convert fail-closed live arm status into a cancellation decision. | PR [#4722](https://github.com/ll7/robot_sf_ll7/pull/4722) added `scripts/dev/watch_live_arm_status.py` and `tests/dev/test_watch_live_arm_status.py`. The watcher is dry-run by default and returns deterministic `scancel` decisions for failed live snapshots or failed event-log rows. | In-repo helper met; deployment proof remains private-operations work. |
| No behavior change for healthy arms; no SLURM submissions from this worker. | PR #4708 tests healthy smoke/full paths. PR #4722 tests completed healthy status returns no cancel command. This audit ran CPU-only validation and did not submit, cancel, merge, release, or delete anything. | Met for repo-side behavior. |

## Remaining Closure Blocker

Issue #4684 should stay open until private operations proves the 15-minute polling cycle is wired to
the live status output and can perform a real `scancel` on a fail-closed arm. The issue comments on
2026-07-07 state this explicitly: PR #4722 completed the in-repository helper, but private-ops
polling-cycle wiring and real cancellation verification are not demonstrable from this repository.

The next empirical action is outside this repository: deploy the watcher in private operations for
the campaign polling cycle, run it against a fail-closed or controlled failure output, and record the
resulting cancellation proof in the private operations state surface. No additional repository code
is required by this audit unless that deployment exposes a mismatch in the emitted status contract.

## Validation

Targeted validation for this audit:

```text
uv run pytest tests/benchmark/test_issue_4367_codesign_campaign_runner.py tests/dev/test_watch_live_arm_status.py -q
```

Expected proof scope: exercises the merged L0-L3 producer contract and the L4 dry-run consumer
contract. It does not prove private-operations deployment or a real SLURM cancellation.
