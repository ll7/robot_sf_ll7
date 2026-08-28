# Goal blocker receipts

Goal blocker receipts make a blocked implementation or execution lane retryable without repeating
the same work. A receipt is a versioned `goal_blocker_receipt.v1` JSON object stored with the active
goal ledger or another private run artifact; it is not a GitHub issue state machine and does not
authorize a maintainer, scientific, compute, merge, or release decision.

The receipt binds the blocker to the inspected issue revision, repository `origin/main`, optional
PR head, dependency/input values, evidence, safe work, next owner, and exact unblock transition.
Its fingerprint preserves the distinction between a missing value, an explicit unavailable value,
an empty string, and `false`.

Use the side-effect-free fingerprint/validation/comparison commands with JSON artifacts:

```bash
uv run python scripts/dev/blocker_receipt.py fingerprint inputs.json
uv run python scripts/dev/blocker_receipt.py validate blocker-receipt.json
uv run python scripts/dev/blocker_receipt.py compare \
  --inputs current-inputs.json --receipt blocker-receipt.json
```

To store a validated receipt in the common-Git active artifact owner (outside the worktree), use
the explicit write command. Without `--path`, it writes
`codex-agent-runs/active/goal-blocker-receipts/issue-<number>.json`:

```bash
uv run python scripts/dev/blocker_receipt.py write blocker-receipt.json
```

`write` is the only mutating command; it replaces the selected small receipt atomically after
validation. The Python API exposes the same boundary through `receipt_artifact_path`,
`write_receipt`, and `load_receipt`.

`blocked_unchanged` suppresses redispatch and reports the recorded owner and unblock condition.
`blocker_changed` re-enters evaluation and names changed fields. Invalid or stale receipts return
`re_evaluate`; they never silently suppress work.

The goal-autopilot snapshot can summarize decision artifacts without reading raw worker logs:

```bash
uv run python -m scripts.dev.autopilot_state_snapshot \
  --blocker-decision blocker-decisions.json --json
```

The snapshot exposes suppressed redispatch counts, re-evaluation counts, blocker classes, and
reasons. All outputs are route evidence; callers still perform fresh issue, branch, dependency, and
validation checks before dispatch or publication.

The canonical candidate-queue snapshot can consume the same external artifact before claim routing;
only its live-admitted `claimable_issues` are claimable:

```bash
uv run python -m scripts.dev.snapshot_issue_batch --claimable \
  --blocker-decision blocker-decisions.json --json
```

An `issue` row with `blocked_unchanged` is classified as `blocked_receipt` and is retained for
audit without entering the autonomous claim queue. `blocker_changed` and `re_evaluate` rows are
classified as `needs_re_evaluation`; they require a fresh admission decision before a worker can
start. Malformed artifacts fail the snapshot closed. The snapshot consumes a decision artifact; the
caller must first run `blocker_receipt.py compare` with fresh issue/body/label, base/head,
dependency, and required-input values.
