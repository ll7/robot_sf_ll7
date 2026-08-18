# Development support tools

This directory contains both automatic readiness gates and intentionally
opt-in support tools. A script is not an automatic PR gate merely because it
lives under `scripts/dev/`; use the invocation surface documented below as the
source of truth.

## Required readiness gates

`BASE_REF=origin/main scripts/dev/pr_ready_check.sh` is the local readiness
entry point for ordinary PR work. CI workflows and the documented PR process
own the required checks that run for every applicable change.

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

## Classification rule

- Required automatic enforcement belongs in an explicit workflow or readiness
  entry point and must have a corresponding contract test.
- Issue-scoped verification requires explicit authoritative arguments and is
  run only from the relevant work packet.
- Repository-administration audits are read-only evidence collectors; they do
  not become merge gates by implication.
