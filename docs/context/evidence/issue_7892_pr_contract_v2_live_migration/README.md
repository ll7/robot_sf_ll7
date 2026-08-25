# PR contract v2 live migration receipt — issue #7892

Schema: `pr_contract_v2_live_migration.v1`

## Inventory

- Repository: `ll7/robot_sf_ll7`
- Audit timestamp: `2026-08-25T09:52:00Z`
- Source commit (inventory tool): `5c78d177f34cb9a18d5ef3e776982054e03dbfd0` (current `origin/main` at slice start)
- Inventory command:
  `python scripts/dev/audit_pr_contract_versions.py --repo ll7/robot_sf_ll7`
- Open-PR counts by classification **before** edit: `v1_compatibility: 8`, `v2_valid: 0`, `v2_invalid: 0`, `body_missing: 0`
- Open-PR counts by classification **after** edit: `v1_compatibility: 7`, `v2_valid: 1`, `v2_invalid: 0`, `body_missing: 0`

## Target PR

- PR: [#7343](https://github.com/ll7/robot_sf_ll7/pull/7343) `fix: fail closed on Robot SF gate evidence`
- Author type: human owner (`ll7`)
- Draft state: draft (unchanged)
- Head SHA: `5b1308eb9d4015232200d99d5f7cd37d4e27e1de` (unchanged before/after edit)
- Changed-path class: workflow/tooling/docs/tests only (`.github/workflows/merge-queue-gate.yml`, `docs/dev_guide.md`, `scripts/dev/merge_queue_gate.py`, `scripts/dev/pr_loop_policy.py`, `scripts/dev/snapshot_pr_queue.py`, `scripts/dev/stacked_prs.py`, `tests/dev/test_merge_queue_gate.py`, `tests/dev/test_pr_loop_policy.py`, `tests/dev/test_stacked_prs.py`)
- Pre-edit body SHA-256: `182c5618a8e458802e61d65f172852a9aa5d5b3d54d64acab36f933d9e159655`
- Post-edit body SHA-256: `6b3912eac767d9f1bd749abeff7124a4102ecf7340fca5ff2a75bab13426f8eb`
- Edit mechanism: REST `PATCH repos/{owner}/{repo}/pulls/7343` (`gh pr edit` is blocked by the retired Projects Classic GraphQL field; REST-only path used)

## Migration

- Prepend-only v2 block; the complete existing human body (Summary / Issue / Validation / Follow-up) is preserved verbatim.
- Declared contract (validated by `parse_pr_contract_v2`):
  - `change_class: tooling`
  - `linked_issues.closes: []`, `relates: [7665, 7892]`
  - `deferred_work.status: open`, `issues: []`, reason records the separately authorized main-protection ruleset change without authorizing it
  - `evidence.applicability: na`, `tier: null`, `result: na`
  - `domain_approval.required: false`, `status: not_required`
  - `performance.claimed: false`
  - `exact_head: 5b1308eb9d4015232200d99d5f7cd37d4e27e1de`
- Local parser result on proposed body before edit: `ok` (canonical `pr_contract_v2.py`)
- Local follow-up analyzer result on proposed body before edit: `ok` ("v2 deferred work has an explicit reason and no follow-up issue")
- Local parser result on live post-edit body: `ok`
- Local follow-up analyzer result on live post-edit body: `ok`

## Live CI observation

- A GitHub `edited` event on a PR does not trigger the repository's CI workflows (workflows trigger on `pull_request` types `labeled`, `unlabeled`, `synchronize`; body edits are not in the trigger set).
- No workflow run was spawned by the body edit; no run IDs to record.
- Canonical remote/body validator result is recorded above in place of hosted-CI coverage; this limitation is stated explicitly and no hosted-CI coverage is claimed.

## Boundaries

- No v1 parser removal or deprecation warning was made.
- No branch-protection or ruleset write was made.
- No mass edit of open PRs was performed.
- No merge or approval of #7343 was performed.
- No benchmark, metric, model, safety, or publication claim is made by this receipt.
- Unavailable observations: hosted CI event for `edited` (not configured); no v1-retirement behavior exists yet.

## Validation commands

```bash
uv run pytest tests/dev/test_audit_pr_contract_versions.py -q        # 14 passed
uv run pytest tests/dev/test_check_pr_followups.py -q               # regression
uv run pytest tests/test_ci_script_contract.py -q                   # regression
uv run ruff check scripts/dev/audit_pr_contract_versions.py tests/dev/test_audit_pr_contract_versions.py
uv run ruff format --check scripts/dev/audit_pr_contract_versions.py tests/dev/test_audit_pr_contract_versions.py
git diff --check
```
