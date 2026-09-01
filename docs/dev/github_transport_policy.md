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
| Pull-request merge | native `gh`, then REST | Exact worktree-base conflict only, with exact-head binding |

The marker classifier is fail-closed: a recognized authentication or permission
marker wins over a generic GraphQL marker, and an unknown error never triggers a
fallback. The policy is transport guidance only; it does not authorize a
benchmark, release, issue closure, or GitHub mutation.

Focused policy tests live in
`tests/dev/test_github_transport_policy.py`. Existing helper-specific tests
remain responsible for endpoint validation and response verification.
