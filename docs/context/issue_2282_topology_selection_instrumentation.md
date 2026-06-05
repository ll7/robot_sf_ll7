# Issue #2282 Topology Selection Instrumentation 2026-06-05

Issue: [#2282](https://github.com/ll7/robot_sf_ll7/issues/2282)
Parent evidence: [issue_2258_topology_primary_route_audit.md](issue_2258_topology_primary_route_audit.md)
Status: instrumentation smoke evidence only.

## Goal

Add compact per-hypothesis score and rejection instrumentation so future topology diagnostics can
explain why a non-primary route hypothesis lost to `primary_route` when alternatives are present.

This closes the evidence gap recorded by Issue #2258/#2267: the prior inventory could count
hypotheses and compare route lengths before scoring, but it could not preserve the score components
or rejection reason for each hypothesis.

## Result

`TopologyGuidedHybridRulePlannerAdapter` now annotates each selectable topology hypothesis with:

- `score_components.length_penalty`
- `score_components.static_clearance_bonus`
- `score_rank`
- `score_margin_to_selected`
- `selection_outcome`
- `rejection_reason`

Rejected selectable alternatives use `lower_topology_selection_score`. The selected hypothesis keeps
`selection_outcome: selected` and `rejection_reason: null`.

The diagnostic trace/report now surfaces compact `topology_selection_score_examples` in the summary
and a Markdown `Topology Selection Score Examples` section. This is instrumentation evidence only;
it does not change topology scoring weights, route proposal behavior, or planner performance claims.

## Smoke Evidence

Reran the Issue #2258 topology diagnostic slice after instrumentation:

- Candidate: `topology_guided_hybrid_rule_v0`
- Scenario: `classic_realworld_double_bottleneck_high`
- Seed: `111`
- Horizon: `160`
- Status: `diagnostic_complete`
- Topology status counts: `{"insufficient_hypotheses": 70, "ok": 90}`
- Selected source counts: `{"dynamic_window": 121, "path_follow_0.5m": 2, "route_guide": 4, "topology_hypothesis": 33}`
- Summary score examples: `10`

The first compact score example is step 1: `primary_route` was selected with score
`-14.828023294990286`, while `masked_cell_71_87` was rejected with
`lower_topology_selection_score` and score margin `0.4686291570846386`.

## Evidence Files

- Compact summary:
  [evidence/issue_2282_topology_selection_instrumentation_2026-06-05/summary.json](evidence/issue_2282_topology_selection_instrumentation_2026-06-05/summary.json)
- Manifest:
  [evidence/issue_2282_topology_selection_instrumentation_2026-06-05/manifest.md](evidence/issue_2282_topology_selection_instrumentation_2026-06-05/manifest.md)
- Source audit:
  [issue_2258_topology_primary_route_audit.md](issue_2258_topology_primary_route_audit.md)

## Validation

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run ruff check robot_sf/planner/topology_guided_local_policy.py scripts/validation/run_topology_hypothesis_diagnostics.py tests/planner/test_topology_guided_local_policy.py tests/validation/test_run_topology_hypothesis_diagnostics.py
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/planner/test_topology_guided_local_policy.py tests/validation/test_run_topology_hypothesis_diagnostics.py
scripts/dev/run_worktree_shared_venv.sh --venv /home/luttkule/git/robot_sf_ll7.worktrees/autopilot-research-cycle-20260605/.venv -- uv run python scripts/validation/run_topology_hypothesis_diagnostics.py --candidate topology_guided_hybrid_rule_v0 --stage full_matrix --scenario-name classic_realworld_double_bottleneck_high --seed 111 --horizon 160 --max-hypotheses 3 --min-hypotheses 2 --output-dir /tmp/robot_sf_issue2282_topology_selection_scores/classic_realworld_double_bottleneck_high_seed111_h160
python - <<'PY'
import json
from pathlib import Path
p = Path("/tmp/robot_sf_issue2282_topology_selection_scores/classic_realworld_double_bottleneck_high_seed111_h160/topology_hypotheses.json")
data = json.loads(p.read_text())
examples = data["summary"]["topology_selection_score_examples"]
assert examples
assert any(item.get("rejection_reason") == "lower_topology_selection_score" for item in examples[0]["hypotheses"])
PY
```

## Claim Boundary

This result proves that the diagnostic path now emits score components, margins, and rejection
reasons on the same topology slice used by the primary-route audit. It is not benchmark-strength
planner evidence and should not be cited as a mitigation or performance improvement.
