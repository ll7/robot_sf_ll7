# Issue #2266 Static-Recenter Activation Diagnostic 2026-06-05

Issue: [#2266](https://github.com/ll7/robot_sf_ll7/issues/2266)
Parent issue: [#2261](https://github.com/ll7/robot_sf_ll7/issues/2261)
Related evidence: [#2221](https://github.com/ll7/robot_sf_ll7/issues/2221)
Status: diagnostic blocker; terminal outcomes are durable, but activation and command-source
evidence is missing.

## Goal

Determine whether static recentering activated during the #2221/#2250 held-out smoke, and whether
activation changed command source, trajectory/progress, or terminal outcome.

This note inspects the tracked compact evidence for the held-out static-recenter transfer smoke:

- Baseline: `hybrid_rule_v3_fast_progress`
- Mechanism row: `issue_2170_static_recenter_only`
- Scenario matrix: `configs/scenarios/sets/issue_2128_heldout_family_transfer_pilot_eval.yaml`
- Seed: `111`
- Horizon: `500`
- Evidence tier: `analysis_only`

It is activation-diagnostic evidence only, not transfer evidence or planner-improvement evidence.

## Finding

The durable #2221 evidence proves terminal outcomes were exactly preserved, but it does not prove
whether static recentering activated. The #2221 manifest states that raw JSONL outputs were used to
derive the tracked summaries and were intentionally not tracked, so activation cannot be
reconstructed from durable artifacts alone.

| Scenario | Seed | Baseline terminal outcome | Recenter terminal outcome | Activation count | First activation step | Command-source changed | Terminal outcome changed | Missing fields |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `classic_station_platform_medium` | 111 | `max_steps`, 60 near misses, 0 collisions | `max_steps`, 60 near misses, 0 collisions | unknown | unknown | unknown | no | activation trace; command source; progress/trajectory delta |
| `francis2023_intersection_wait` | 111 | success, 0 near misses, 0 collisions | success, 0 near misses, 0 collisions | unknown | unknown | unknown | no | activation trace; command source; progress/trajectory delta |

## Interpretation

The #2221 smoke supports the published `slice_local` terminal-metric classification, but it does
not distinguish these mechanisms:

- static recentering never activated;
- static recentering activated too late or in irrelevant states;
- static recentering activated but lost scoring arbitration;
- static recentering changed a micro command without changing terminal outcome;
- or the held-out scenarios did not contain the failure mode recentering was meant to address.

Because activation and command-source data are not durable, the next #2261 step should not tune or
retire static recentering based on activation assumptions. The smallest useful follow-up is to
locate the untracked run outputs named in the manifest or run an instrumented trace-level rerun for
the same held-out smoke that records activation count, first activation step, selected command
source, and progress/trajectory delta.

## Evidence

- Prior transfer note:
  [issue_2221_static_recenter_transfer.md](issue_2221_static_recenter_transfer.md)
- Prior compact comparison:
  [evidence/issue_2221_static_recenter_transfer_2026-06-04/comparison_summary.json](evidence/issue_2221_static_recenter_transfer_2026-06-04/comparison_summary.json)
- Prior manifest:
  [evidence/issue_2221_static_recenter_transfer_2026-06-04/manifest.md](evidence/issue_2221_static_recenter_transfer_2026-06-04/manifest.md)
- Baseline report:
  [policy_search/reports/2026-06-04_hybrid_rule_v3_fast_progress_full_matrix.md](policy_search/reports/2026-06-04_hybrid_rule_v3_fast_progress_full_matrix.md)
- Mechanism report:
  [policy_search/reports/2026-06-04_issue_2170_static_recenter_only_full_matrix.md](policy_search/reports/2026-06-04_issue_2170_static_recenter_only_full_matrix.md)
- This compact summary:
  [evidence/issue_2266_static_recenter_activation_2026-06-05/summary.json](evidence/issue_2266_static_recenter_activation_2026-06-05/summary.json)
- This activation table:
  [evidence/issue_2266_static_recenter_activation_2026-06-05/activation_table.csv](evidence/issue_2266_static_recenter_activation_2026-06-05/activation_table.csv)
- This manifest:
  [evidence/issue_2266_static_recenter_activation_2026-06-05/manifest.md](evidence/issue_2266_static_recenter_activation_2026-06-05/manifest.md)

## Validation

This analysis used tracked compact evidence only. No new benchmark or broader rerun was performed.

Validation commands:

```bash
git show origin/main:docs/context/evidence/issue_2221_static_recenter_transfer_2026-06-04/comparison_summary.json | jq 'keys'
python -m json.tool docs/context/evidence/issue_2266_static_recenter_activation_2026-06-05/summary.json
python - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(Path("docs/context/evidence/issue_2266_static_recenter_activation_2026-06-05/activation_table.csv").open()))
assert len(rows) == 2
assert all(row["activation_count"] == "unknown" for row in rows)
assert all(row["terminal_outcome_changed"] == "false" for row in rows)
PY
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
