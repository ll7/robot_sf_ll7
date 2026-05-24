# Issue #1477 S10/H500 Mechanism Review Evidence (2026-05-24)

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1477>

## Source

This bundle reviews selected cells from the issue #1454 S10/h500 candidate campaign that PR #1463
summarizes for issue #1462.

The source archive was verified before selection:

```text
44ec1d4eb89d450eb204398a3807185ce9bdd4aae0eeb5e55af0704fd4a8b0fc  issue1454-s10-h500-candidates-2026-05-23.tar.zst
```

The available archive contains per-episode `episodes.jsonl` summaries. It does not contain step
traces, rendered videos, trajectory JSONL, or frame artifacts for the selected cells.

## Reproduction

```bash
python3 scripts/tools/review_issue_1477_s10_h500_mechanisms.py \
  --raw-campaign-dir <verified extraction>/issue1454-s10-h500-candidates \
  --archive <verified local archive>/issue1454-s10-h500-candidates-2026-05-23.tar.zst \
  --expected-archive-sha256 44ec1d4eb89d450eb204398a3807185ce9bdd4aae0eeb5e55af0704fd4a8b0fc \
  --output-dir docs/context/evidence/issue_1477_s10_h500_mechanism_review_2026-05-24 \
  --per-scenario 2
```

## Files

- `reviewed_cells.csv` - exact planner/scenario/seed/episode rows selected for the mechanism audit.
- `reviewed_cells.md` - compact human-readable version of the selected rows.
- `summary.json` - archive checksum, target scenarios, candidate planner set, and review status.

## Boundary

This evidence supports only summary-level statements: exact outcomes, collisions, near-miss counts,
minimum clearing distance, and time-to-goal normalization for selected cells. It does not support
causal mechanism language such as waiting, yielding, hesitation, squeezing through crowds, or
intentional risk-taking because the source archive lacks step traces and videos.
