# Issue #1477 S10/H500 Mechanism Review

Date: 2026-05-24

## Goal

Review a small set of S10/h500 candidate campaign cells from issue #1454 / PR #1463 to determine
whether the aggregate failure-mode tables can support behavioral mechanism claims for the hardest
candidate weak spots and near-miss-heavy candidate wins.

## Inputs

- Source aggregate note:
  [`issue_1462_s10_h500_failure_modes.md`](issue_1462_s10_h500_failure_modes.md)
- Source compact evidence:
  [`evidence/issue_1462_s10_h500_failure_modes_2026-05-24/README.md`](evidence/issue_1462_s10_h500_failure_modes_2026-05-24/README.md)
- Raw archive checksum:
  `44ec1d4eb89d450eb204398a3807185ce9bdd4aae0eeb5e55af0704fd4a8b0fc`
- Review evidence:
  [`evidence/issue_1477_s10_h500_mechanism_review_2026-05-24/README.md`](evidence/issue_1477_s10_h500_mechanism_review_2026-05-24/README.md)

## Method

The review used the existing `episodes.jsonl` summaries from the verified issue #1454 archive. The
selector prioritized two candidate cells per target scenario:

- weak spots: `francis2023_narrow_doorway`, `classic_station_platform_medium`,
- near-miss-heavy wins: `francis2023_robot_crowding`, `francis2023_narrow_hallway`,
- optional largest success gain: `classic_bottleneck_high`.

The command was:

```bash
python3 scripts/tools/review_issue_1477_s10_h500_mechanisms.py \
  --raw-campaign-dir <verified extraction>/issue1454-s10-h500-candidates \
  --archive <verified local archive>/issue1454-s10-h500-candidates-2026-05-23.tar.zst \
  --expected-archive-sha256 44ec1d4eb89d450eb204398a3807185ce9bdd4aae0eeb5e55af0704fd4a8b0fc \
  --output-dir docs/context/evidence/issue_1477_s10_h500_mechanism_review_2026-05-24 \
  --per-scenario 2
```

## Findings

The selected weak-spot rows reproduce the aggregate boundary from issue #1462:
`francis2023_narrow_doorway` candidate rows terminate without collision, near misses, or success,
while `classic_station_platform_medium` candidate rows terminate with high near-miss exposure
(`158` near misses for seed `114` in both selected static-escape variants).

The selected near-miss-heavy wins show successful candidate rows with substantial near-miss counts:
`francis2023_robot_crowding` reaches `281` and `265` near misses in selected successful continuous
static-escape rows, and `francis2023_narrow_hallway` reaches `116` near misses in two selected
successful rows. The optional `classic_bottleneck_high` cells are also successful with elevated
near-miss counts (`121` and `108`).

## Interpretation

The review does not support causal mechanism claims. The verified archive contains episode-level
summary records but no step traces, videos, trajectory files, or frame artifacts. The selected cells
can support cautious summary language such as "near-miss-heavy successful rows" or "terminated
weak-spot rows"; they cannot justify behavioral wording such as waiting, yielding, hesitation,
squeezing through crowds, or intentional risk-taking.

## Recommendation

Keep the issue #1462 interpretation boundary: mechanism claims remain trace-required. Paper-facing
or discussion-facing text should describe the S10/h500 candidate results as aggregate
success/collision/near-miss patterns unless a future run preserves step traces or rendered videos
for the exact cells in `reviewed_cells.csv`.
