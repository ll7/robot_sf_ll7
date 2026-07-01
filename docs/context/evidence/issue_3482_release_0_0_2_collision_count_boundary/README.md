# Issue #3482 Release 0.0.2 Collision-Count Boundary

This bundle records the current release `0.0.2` collision-count claim boundary after
`EpisodeEventLedger.v1` exact/derived reconciliation surfaced a discrepancy and the 2026-07-01
recovery passes did not find exact-event provenance artifacts.

The public release bundle supports the published derived side of the mismatch:

- release `0.0.2` rows: `987`
- derived `total_collision_count > 0` rows: `0`

It does not prove the diagnostic exact-event side:

- exact collision outcomes: `241`
- reconciliation violations: `241`

## Contents

| File | Purpose |
| --- | --- |
| `manifest.json` | Machine-readable boundary: release `0.0.2` collision-count claims are withdrawn because exact-event provenance is unavailable. |

## Validation

```bash
uv run python scripts/validation/check_release_0_0_2_collision_count_boundary.py --json
```

The checker fails closed if the manifest marks release `0.0.2` collision-count metrics as
paper-ready or keeps promotion gates open after the withdrawal decision.

## Scope Boundary

This is not a full benchmark campaign run, not a Slurm/GPU submission, not a committed durable
`0.0.2` reconciliation bundle, and not evidence that the collision-count metric was valid. It is a
claim-withdrawal boundary: non-collision release table fields may remain usable with caveats, but
release `0.0.2` collision-count-derived claims must not be used for paper/dissertation evidence
unless future exact-event provenance is recovered and promoted.
