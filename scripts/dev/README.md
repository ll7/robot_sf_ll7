# Development support tools

This directory contains both automatic readiness gates and intentionally
opt-in support tools. A script is not an automatic PR gate merely because it
lives under `scripts/dev/`; use the invocation surface documented below as the
source of truth.

## Required readiness gates

`BASE_REF=origin/main scripts/dev/pr_ready_check.sh` is the local readiness
entry point for ordinary PR work. CI workflows and the documented PR process
own the required checks that run for every applicable change.

If readiness receives a termination signal, it preserves a private,
bounded receipt under `output/validation/pr_ready/` with the active phase,
lane, process-group cleanup result, and small host/cgroup resource snapshot.
Set `PR_READY_TERMINATION_RECEIPT=/absolute/path/receipt.json` when a caller
needs a stable handoff path. The receipt never includes the command line or
environment, and readiness still returns the conventional signal status (143
for `SIGTERM`, the termination signal).

The native merge queue enforcement path is
[`merge_queue_gate.py`](merge_queue_gate.py), invoked by
[`.github/workflows/merge-queue-gate.yml`](../../.github/workflows/merge-queue-gate.yml)
on `merge_group`. The standalone protection audit below does not replace that
workflow or change branch protection.

## Explicit issue-scoped verification

[`check_pr_closing_reference.py`](check_pr_closing_reference.py) verifies that
GitHub's GraphQL `closingIssuesReferences` for a specific PR contains an issue
number supplied by an already authorized work packet. It must not infer the
expected issue from PR prose.

```bash
uv run python scripts/dev/check_pr_closing_reference.py \
  <pr-number> <expected-issue-number> --repo ll7/robot_sf_ll7
```

Add `--json` for the machine-readable result. Exit status is `0` when the
expected issue is present, `1` when it is absent, and `2` when the result is
unavailable or the input is invalid. The command is read-only; it does not
edit PR bodies, labels, issues, or branches. Because it queries GitHub, an
API failure is an unknown result and must remain a blocker.

## Repository-administration audits

[`check_merge_queue_protection.py`](check_merge_queue_protection.py) is a
read-only audit for the maintainer-owned merge-queue activation dimensions
from issue #6404. Run its deterministic offline contract test with:

```bash
uv run python scripts/dev/check_merge_queue_protection.py --self-test
```

After a maintainer changes branch/ruleset settings, run the live audit with:

```bash
uv run python scripts/dev/check_merge_queue_protection.py \
  --check --repo ll7/robot_sf_ll7 [--pr <enqueued-pr-number>]
```

`--pr` is optional and is only meaningful for a PR that is actually enqueued;
it probes the live `ALLGREEN` strategy. The audit fails closed when a required
dimension is unsatisfied or unverifiable. It performs no ruleset, branch,
queue, PR, issue, or workflow mutation, and it cannot claim that a real
`merge_group` run exists unless GitHub provides that evidence.

## CI inline-logic helpers

The CI aggregate workflow extracts its reusable executable logic into tested
helpers (issue #7666):

- [`model_cache_key.py`](model_cache_key.py) derives the exact-repeat model-cache
  key from registry-pinned digests:

  ```bash
  uv run python scripts/dev/model_cache_key.py --config <ppo-config.yaml> --machine
  ```

- [`merge_test_durations.py`](merge_test_durations.py) validates and merges the four
  pytest-split duration shard stores:

  ```bash
  uv run python scripts/dev/merge_test_durations.py \
    --artifact-dir .duration-artifacts --output .test_durations
  ```

- [`check_ci_needs.py`](check_ci_needs.py) evaluates the aggregate job's required
  needs results with event-specific coverage rules:

  ```bash
  uv run python scripts/dev/check_ci_needs.py --event-name pull_request --results '{"fast-feedback": "success"}'
  ```

Focused tests live in `tests/dev/test_ci_helpers.py`; the workflow-contract parity
checks live in `tests/test_ci_script_contract.py`.

## Open-issue goal-autopilot preparation

[`prepare_open_issue_contracts.py`](prepare_open_issue_contracts.py) consumes the
report-only `open_issue_contract_audit.v1` output from
[`audit_open_issue_contracts.py`](audit_open_issue_contracts.py) and emits per-issue
`goal-autopilot` preparation packets (issue #7929). Plan, render, and verify modes
are report-only; apply requires an explicit reviewed plan and a bounded, CAS-guarded
batch. Real apply additionally requires the exact selected issue list and its
`content_sha256` passed as `--reviewed-plan-digest`.

```bash
uv run python scripts/dev/prepare_open_issue_contracts.py \
  --audit-json /tmp/open_issue_audit.json \
  --plan-json /tmp/open_issue_preparation_plan.json \
  --batch-id <stable-batch-id>
```

See [`docs/ai/open-issue-contract-preparation.md`](../../docs/ai/open-issue-contract-preparation.md)
for the operator contract. Focused offline tests live in
`tests/dev/test_prepare_open_issue_contracts.py`.

## Parent goal-autopilot arbitration

[`goal_autopilot_controller.py`](goal_autopilot_controller.py) is the
machine-checked parent arbiter for the continuous implement/review/merge/
discover loop. Child workers may report only lane-local exhaustion, such as
`implementation_queue_exhausted`; only the parent can emit
`genuine_zero_work`. A terminal result includes a fresh
`goal_autopilot_zero_work_proof.v1` receipt bound to the origin/main SHA,
issue/claim state, PR heads, preparation audit, and discovery inputs.

```bash
uv run python scripts/dev/goal_autopilot_controller.py \
  --snapshot /tmp/goal_autopilot_controller_snapshot.json --json
```

The arbiter routes merge, review, recovery, implementation, readiness-gate,
formalization, and discovery work before considering a terminal result. Focused
regression coverage lives in `tests/dev/test_goal_autopilot_controller.py`.

## Classification rule

- Required automatic enforcement belongs in an explicit workflow or readiness
  entry point and must have a corresponding contract test.
- Issue-scoped verification requires explicit authoritative arguments and is
  run only from the relevant work packet.
- Repository-administration audits are read-only evidence collectors; they do
  not become merge gates by implication.

## Delegated-worker worktree guard

`worktree_receipt.py` is the opt-in pre-write boundary for repository-owned delegated workers.
Pair `create_worktree.sh --receipt PATH --task-id ID` with `--exec`; creation writes an atomic
receipt and checks it before launching the command. The check is read-only, fail-closed, and
machine-readable. It validates the current working directory, assigned absolute worktree, linked
Git common directory, branch/ref, and base ancestry. Human callers that omit receipt options retain
the ordinary path.
