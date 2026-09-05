# GitHub Helper Transport Policy

The repository GitHub helpers use one fail-closed transport policy. It keeps
native `gh` reads narrowly eligible for known GraphQL-path failures, while
authentication, authorization, repository-resolution, malformed-response, and
write-verification failures remain errors.

## Discover the policy

From the repository root, show the registered contracts with:

```bash
uv run python scripts/dev/github_transport_policy.py show
```

Audit every issue, pull-request, and comment helper before changing the helper
set:

```bash
uv run python scripts/dev/github_transport_policy.py audit --json
```

The audit discovers `scripts/dev/gh_*.py` and `scripts/dev/gh_*.sh`. A new
helper must be added to the registry in
`scripts/dev/github_transport_policy.py`, reference the policy at runtime, and
name a focused smoke-test path. Missing registration or proof fails the audit.

## Current routes

| Helper family | Allowed route | Fallback boundary |
| --- | --- | --- |
| Issue thread reader | native `gh`, then REST | Known GraphQL/deprecated-field errors only |
| Issue-view compatibility wrapper | native `gh` via shared reader, then REST (comments mode is REST-only) | Same known GraphQL-path errors as the issue reader |
| Issue and pull-request comments | REST | None |
| Pull-request body, labels, comments, and reviews | REST | None |
| Pull-request merge | single-account receipt-owner delegation | No shell/native/REST merge fallback; the owner performs the exact-head write |

The marker classifier is fail-closed: a recognized authentication or permission
marker wins over a generic GraphQL marker, and an unknown error never triggers a
fallback. The policy is transport guidance only; it does not authorize a
benchmark, release, issue closure, or GitHub mutation.

Focused policy tests live in
`tests/dev/test_github_transport_policy.py`. Existing helper-specific tests
remain responsible for endpoint validation and response verification.

The compatibility helper `scripts/dev/gh_pr_merge.sh` is a receipt-owner
caller. It validates its positive PR number, full expected head, repository
identity, and transport-policy admission, then delegates report and apply to
`scripts/dev/single_account_merge_receipt.py`. It has no merge transport of
its own, so a worktree conflict, quota problem, or other transport failure
cannot select an unguarded writer. Ordinary-CAS evidence and the owner’s
immediate live recheck remain authoritative.

The helper deliberately does not delete the source branch. Source-branch
cleanup is a separate guarded post-merge action and remains subject to the
`gh-pr-merger` condition that no unique, unpreserved work remains. Invoking the
compatibility helper alone does not authorize that cleanup.

The authority fixture also declares the scan surfaces for runtime helpers,
workflow YAML, documentation, and skill guidance. Its self-check rejects
direct merge endpoints, split shell REST mutations, and native pull-request
merge commands outside the receipt owner.
