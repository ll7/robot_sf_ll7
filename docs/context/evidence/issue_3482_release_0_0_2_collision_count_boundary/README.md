# Issue #3482 Release 0.0.2 Collision-Count Boundary

This bundle records the current release `0.0.2` collision-count claim boundary after
`EpisodeEventLedger.v1` exact/derived reconciliation surfaced a discrepancy: exact collision
outcomes are present, but the derived collision-count metric is not usable as paper-ready evidence
until the durable reconciliation bundle is promoted and affected tables are annotated.

## Contents

| File | Purpose |
| --- | --- |
| `manifest.json` | Machine-readable boundary: exact outcomes remain usable with the ledger boundary, while release `0.0.2` collision-count metrics stay blocked. |

## Validation

```bash
uv run python scripts/validation/check_release_0_0_2_collision_count_boundary.py --json
```

The checker fails closed if the manifest marks the release `0.0.2` collision-count metric as
claim-ready while the documented exact/derived discrepancy and open gates remain.

## Scope Boundary

This is not a full benchmark campaign run, not a Slurm/GPU submission, not the committed durable
`0.0.2` reconciliation bundle, and not a paper or dissertation claim edit.
