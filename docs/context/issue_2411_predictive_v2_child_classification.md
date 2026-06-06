# Issue #2411 Predictive-v2 Child Classification

Date: 2026-06-06
Issue: <https://github.com/ll7/robot_sf_ll7/issues/2411>
Parent: <https://github.com/ll7/robot_sf_ll7/issues/1490>
Prior decision: [issue_2275_predictive_v2_fate.md](issue_2275_predictive_v2_fate.md)

## Scope

This note refreshes the predictive-v2 child issue states after the Issue 2275 stop/revise
decision. It does not run predictive-v2 comparisons, train models, submit SLURM jobs, or claim a
new planner result.

The compact classification artifacts live under
`docs/context/evidence/issue_2411_predictive_v2_child_classification_2026-06-06/`.

## Decision Basis

- Issue 1543 found the obstacle-feature prerequisite negative for closed-loop transfer:
  predictive success moved from `0.1304` to `0.1014`, and hard-seed success stayed `0.0000` for
  both variants.
- Issue 1897 then failed the local planner-side coupling gate: `baseline_like` and
  `phase_coupled_sequence_gate` both recorded global success `0.0000` and hard success `0.0000`;
  the revised row only improved global mean min-distance by `0.0108`.
- Issue 2275 therefore selected `stop_old_predictive_v2_expansion`; future predictive-v2 work must
  name a new planner-coupling or planner-aligned objective hypothesis and pass a bounded
  closed-loop gate before reopening expansion execution.

## Child Issue Table

| Issue | Current state checked | Classification | Recommendation | Evidence |
| --- | --- | --- | --- | --- |
| Issue 1505 | open, `state:blocked`, `evidence:proposal`, `resource:local` | close | Close the old data-row preflight path, or leave blocked only with a comment that it is superseded by Issue 2275/Issue 2411. | It exists to preflight the old four-way rows, but Issue 1897 failed the prerequisite gate and Issue 2275 stopped the old expansion. |
| Issue 1506 | open, `state:blocked`, `evidence:proposal`, `resource:slurm` | close | Close the old four-way Slurm matrix path. | There is no passing local preflight to justify Slurm spend, and running it would contradict Issue 2275. |
| Issue 1507 | open, `state:blocked`, `evidence:proposal`, `resource:local` | narrow | Narrow only to closeout/downgrade synthesis if kept open; otherwise close with Issue 2275/Issue 2411 as the evidence record. | The original forecast-to-control transfer analysis depends on new comparable evidence that the stopped Issue 1506 path will not produce. |
| Issue 1490 | open parent, `state:blocked`, `evidence:proposal`, `resource:slurm` | decision-required | Maintainer should choose whether to close the parent or keep it as a blocked umbrella with updated stop/revise language and no executable children. | The parent already records the failed Issue 1897 gate, but its lifecycle remains open and can still attract stale execution attempts. |
| Issue 2275 | closed | keep | Keep as the canonical stop/revise decision. | PR 2287 implemented the fate decision and decision matrix. |

## Recommended GitHub Follow-Up

After this classification is accepted, post short comments on Issues 1505, 1506, and 1507 linking
this note and Issue 2275. Close Issues 1505 and 1506 unless the maintainer prefers to keep blocked
historical records open. Narrow or close Issue 1507 depending on whether a bounded predictive-v2
closeout synthesis is still desired. Leave Issue 1490 open only if its body/labels continue to make
the stop/revise boundary obvious.

## Claim Boundary

This is issue-routing synthesis. It prevents negative and failed preflight evidence from being
treated as benchmark-ready execution work, but it is not a new benchmark, training, or paper-facing
planner result.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_2411_predictive_v2_child_classification_2026-06-06/summary.json
python - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(Path(
    "docs/context/evidence/issue_2411_predictive_v2_child_classification_2026-06-06/child_issue_classification.csv"
).open(newline="")))
assert {row["issue"] for row in rows} == {"1490", "1505", "1506", "1507", "2275"}
assert {row["classification"] for row in rows} <= {
    "keep",
    "narrow",
    "close",
    "supersede",
    "decision-required",
}
PY
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_2411_predictive_v2_child_classification.md \
  --path docs/context/evidence/issue_2411_predictive_v2_child_classification_2026-06-06/summary.json \
  --path docs/context/catalog.yaml \
  --path docs/context/evidence/README.md
git diff --check
```
