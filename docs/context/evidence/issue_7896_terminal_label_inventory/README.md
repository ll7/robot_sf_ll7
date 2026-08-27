# Terminal-Label Inventory — issue #7896

**Status:** report-only inventory implementation delivered; the tracked live receipt is an
explicitly bounded diagnostic sample and applied zero label mutations.
**Issue:** [#7896](https://github.com/ll7/robot_sf_ll7/issues/7896) (relates #7651, #7609).
**Owner module:** `scripts/dev/terminal_label_reconcile.py` (inventory mode).

## Policy changes

- `agent` is now an active dispatch label and is removed by the plan for verified terminal
  classes (alongside `agent-ready`, `merge-ready`, `state:ready`, `state:running`, `state:working`,
  `state:review`, `needs-review`, `needs-triage`, dependency holds, and `bounty:in-progress`).
- Every known `state:*` qualifier is explicitly classified (`active` / `terminal` / `historical`);
  unknown `state:*` labels fail closed as ambiguous — no unreviewed wildcard removal.
- Reopened and `terminal_unverified` items retain active labels and receive no false terminal plan.

## Inventory mode

```bash
uv run python scripts/dev/terminal_label_reconcile.py \
  --inventory terminal --repo ll7/robot_sf_ll7 \
  --report output/terminal_label_plan.json [--max-items N] [--max-pages N]
```

- Pages all closed issues and PRs (page-based REST; bounded via `--max-items` when recorded).
- Re-reads every item through the canonical exact-item owner before producing its plan and compares
  the paginated row with the exact reread for terminal class and labels; drift is non-applicable.
- Fails closed on truncation, duplicate numbers, unknown state reasons, malformed labels, REST or
  rate-limit errors, inconsistent rereads, and ambiguous `state:*` qualifiers.
- `mutation_authorized: false`; no inventory-wide `--apply` path exists.
- Deterministic JSON apart from the isolated observation timestamp; repeated runs over a fixed
  fixture are byte-stable. Reports expose source endpoints, pagination completeness, per-item
  verdicts, and aggregate counts by terminal class, label, item kind, and verdict.

## Live bounded receipt (200 most-recent closed items)

- Observation: `2026-08-27T14:01:49.629706+00:00`; 2 pages, 200 rows, `complete: false` because
  the explicit `--max-items 200` bound stopped pagination. This is diagnostic-only route evidence,
  not a repository-wide exact mutation count.
- 82 `completed` + 113 `pr_merged` + 5 `pr_closed_unmerged` rows.
- Candidate label removals: 7 `agent`, 1 `dependency:has-blockers`, 100 `merge-ready`, and
  1 `needs-review` (zero applied).
- Full-report SHA-256: `275b8d4acca8ba65aac9f1dcca03776d431e6c365bf622acda89129f96e9382e`
  (`output/terminal_label_plan.json`, ignored/worktree-local).
- Repository commit: `a7158eb1e8a4782a95b10fa499b4d3b4bc9b3193`; source API: `github-rest-v3`.
- Zero-mutation proof: inventory mode never calls `add_label`/`remove_label`
  (`test_inventory_zero_mutation_proof`); malformed, error, truncation, and reread-drift fixtures
  are covered by the focused suite.

## Boundary

No bulk label apply, closure, reopen, assignment, comment, review, merge, or branch-protection
change. Evidence/resource/type/priority/domain-review/provenance labels are preserved. A separate
apply issue is required to execute any plan.
