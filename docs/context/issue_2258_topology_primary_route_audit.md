# Issue #2258 Topology Primary-Route Audit 2026-06-05

Issue: [#2258](https://github.com/ll7/robot_sf_ll7/issues/2258)
Date: 2026-06-05
Status: diagnostic-only analysis; topology alternatives exist, but primary-route dominance remains.

## Goal

Explain why the #2223/#2251 topology-guided diagnostic selected only `primary_route`, switched 0
times, and produced explanation signal without mitigation.

This note is a trace-mechanism review over one diagnostic slice:

- Candidate: `topology_guided_hybrid_rule_v0`
- Scenario: `classic_realworld_double_bottleneck_high`
- Seed: `111`
- Horizon: `160`
- Evidence tier: `diagnostic_trace_smoke`

It is not benchmark-strength planner-improvement evidence.

## Finding

The primary cause is not total absence of alternatives. The regenerated trace exposed alternative
hypotheses on 90 of 160 steps:

- 70 steps were classified `insufficient_hypotheses`: 52 logged zero hypotheses and 18 logged one.
- 55 `ok` steps had two hypotheses.
- 35 `ok` steps had three hypotheses.

However, the selected topology hypothesis stayed `primary_route`:

- 33 total topology-command wins selected `primary_route`.
- 28 of those wins occurred on `ok` steps with at least one alternative present.
- 5 of those wins occurred on insufficient-hypothesis steps 92-96 where only `primary_route` was
  available.
- The explicit hypothesis switch count was 0.

Among all 90 `ok` steps, the primary route was shorter than or tied with the best alternative on
88 steps. During the 28 `ok` topology-command wins, the primary route was shorter or tied on 27
steps. The only `ok` topology-command win with a shorter alternative was step 159, where the
selected hypothesis still remained `primary_route`.

## Mechanism Interpretation

The diagnostic route proposer is producing alternatives often enough to reject the strongest
"alternatives were absent" explanation. The failure is more specific:

1. The masking heuristic usually finds alternatives that are longer than the primary route.
2. Static-clearance minima are similar across the primary and masked alternatives in this slice, so
   the configured route score has little reason to prefer an alternate path.
3. The downstream topology command can win safety scoring, but it inherits the selected hypothesis;
   because selection stays on `primary_route`, command wins do not become corrective reroutes.
4. The current diagnostic does not log enough per-hypothesis selection-score detail to fully explain
   the one step where an alternative had a shorter remaining route but primary still selected.

## Recommendation

Revise topology hypothesis generation before retuning downstream planner scoring.

The next useful implementation should expose or change the upstream hypothesis selection surface:

- persist per-hypothesis selection scores in compact diagnostics;
- add a diversity/commitment term that can prefer a non-primary homotopy when it is meaningfully
  safer or less stalled;
- or create a targeted falsification run where a known non-primary route should win.

Do not promote `topology_guided_hybrid_rule_v0` as mitigation evidence from this slice. It remains
useful as an explanation/trace probe.

## Evidence

- Prior diagnostic note:
  [issue_2223_topology_hypothesis_planning.md](issue_2223_topology_hypothesis_planning.md)
- Prior compact evidence:
  [evidence/issue_2223_topology_hypothesis_2026-06-04/summary.json](evidence/issue_2223_topology_hypothesis_2026-06-04/summary.json)
- This audit summary:
  [evidence/issue_2258_topology_primary_route_audit_2026-06-05/summary.json](evidence/issue_2258_topology_primary_route_audit_2026-06-05/summary.json)
- This pre-scoring inventory:
  [evidence/issue_2258_topology_primary_route_audit_2026-06-05/topology_hypothesis_inventory.csv](evidence/issue_2258_topology_primary_route_audit_2026-06-05/topology_hypothesis_inventory.csv)
- This audit manifest:
  [evidence/issue_2258_topology_primary_route_audit_2026-06-05/manifest.md](evidence/issue_2258_topology_primary_route_audit_2026-06-05/manifest.md)
- Candidate config:
  [../../configs/policy_search/candidates/topology_guided_hybrid_rule_v0.yaml](../../configs/policy_search/candidates/topology_guided_hybrid_rule_v0.yaml)
- Topology policy implementation:
  [../../robot_sf/planner/topology_guided_local_policy.py](../../robot_sf/planner/topology_guided_local_policy.py)

## Validation

Regenerated the diagnostic trace in an isolated worktree with the shared virtualenv:

```bash
scripts/dev/run_worktree_shared_venv.sh \
  --venv /home/luttkule/git/robot_sf_ll7.worktrees/autopilot-research-cycle-20260605/.venv \
  -- uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir output/diagnostics/issue2258_topology_audit/classic_realworld_double_bottleneck_high_seed111_h160
```

Raw regenerated traces remain under ignored `output/`. Only the compact summary and manifest are
tracked, along with the generated per-step hypothesis inventory CSV.
