# Terminal-Label Inventory — issue #7896

**Status:** report-only inventory delivered; zero label mutations applied.
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
  --report output/terminal_label_plan.json [--max-items N]
```

- Pages all closed issues and PRs (page-based REST; bounded via `--max-items` when recorded).
- Re-reads every item through the canonical exact-item owner before producing its plan.
- Fails closed on truncation, duplicate numbers, unknown state reasons, malformed labels, REST
  errors, and ambiguous `state:*` qualifiers.
- `mutation_authorized: false`; no inventory-wide `--apply` path exists.
- Deterministic JSON apart from the isolated observation timestamp; repeated runs over a fixed
  fixture are byte-stable.

## Live bounded receipt (200 most-recent closed items)

- 85 `completed` + 115 `pr_closed_unmerged` rows.
- Candidate label removals: 6 `agent`, 94 `merge-ready`, 5 `needs-review` (zero applied).
- Full-report SHA-256: `8e49fc670ae2286ed4bf4f69de83a75c97aad99fa8322725a5ea94dc3427d4dd`
  (`output/terminal_label_plan.json`, ignored/worktree-local).
- Zero-mutation proof: inventory mode never calls `add_label`/`remove_label`
  (`test_inventory_zero_mutation_proof`).

## Boundary

No bulk label apply, closure, reopen, assignment, comment, review, merge, or branch-protection
change. Evidence/resource/type/priority/domain-review/provenance labels are preserved. A separate
apply issue is required to execute any plan.
