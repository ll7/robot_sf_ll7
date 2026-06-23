# Issue 3472: PR Body Contract Workflow

Issue: [#3472](https://github.com/ll7/robot_sf_ll7/issues/3472)

## Summary

This change adds a lightweight `pull_request` workflow that runs the existing PR body contract
checker against the live GitHub event body. The workflow also fetches the PR changed-file list so
the checker can reject empty or bot-only descriptions for substantive source/configuration changes
without blocking docs-only/template-only PRs on the same rule.

## Contract

- `scripts/dev/check_pr_followups.py --require-body` fails closed when the PR body is unavailable.
- `--require-open-issues` verifies linked follow-up issues for declared deferred work are open.
- Domain-aware approval remains required when Research Result Guidance contains non-`NA`/
  non-`docs-only` evidence-result fields.
- `--require-substantive-body --changed-files-file <file>` rejects empty or bot-only bodies only
  when changed files include substantive source/configuration paths.

## Regression Fixtures

The focused tests model the June 2026 audit failures:

- PRs #3414 and #3415: empty PR bodies on substantive benchmark-code refactors.
- PR #3416: CodeRabbit-only release notes on substantive benchmark-code refactor.
- PRs #3449 and #3450: evidence-sensitive PR bodies that predate the current structured
  `Domain-Aware Approval` and `Follow-Up Issues` contract.

## Validation

- `./scripts/dev/run_worktree_shared_venv.sh -- pytest tests/dev/test_check_pr_followups.py tests/test_ci_script_contract.py -q`
- `./scripts/dev/run_worktree_shared_venv.sh -- ruff check scripts/dev/check_pr_followups.py tests/dev/test_check_pr_followups.py tests/test_ci_script_contract.py`

## Claim Boundary

This is a repository workflow guard, not a replacement for human review or branch-protection
configuration. It makes the contract enforceable in CI and suitable for later branch-protection
adoption once the new workflow has run on live PRs.
