# Issue #2223 Topology-Hypothesis Planning Diagnostic

Issue: [#2223](https://github.com/ll7/robot_sf_ll7/issues/2223)
Date: 2026-06-04
Status: diagnostic trace smoke; explanation signal without mitigation.

## Goal

Test whether topology-hypothesis local route alternatives explain or mitigate constrained-scene
deadlock behavior on a bottleneck slice.

This is diagnostic-only evidence for `topology_guided_hybrid_rule_v0`, whose registry
`claim_scope` is `diagnostic_only`. It is not benchmark-strength planner improvement evidence.

## Result

The topology-guided candidate exposed alternate-route diagnostics and selected topology commands,
but it did not improve the matched h160 progress outcome.

| Row | Selected command sources | Net progress | Final goal distance | Closest robot-ped distance | Collision flags |
| --- | --- | ---: | ---: | ---: | --- |
| `hybrid_rule_v3_fast_progress` | `dynamic_window: 158`, `path_follow_0.5m: 2` | -12.7885 m | 14.7885 m | 1.8982 m | 0 |
| `topology_guided_hybrid_rule_v0` | `dynamic_window: 121`, `path_follow_0.5m: 2`, `route_guide: 4`, `topology_hypothesis: 33` | -12.8258 m | 14.8258 m | 1.9440 m | 0 |

Topology trace summary:

- topology alternatives were sufficient on 90 of 160 steps;
- topology commands won 33 steps;
- every topology-command win selected `primary_route`;
- explicit `hypothesis_switch_count` was 0;
- no fallback count or collision flags were observed.

## Classification

Classification: `explanation_signal_without_mitigation`.

The trace supports the weaker claim that topology hypotheses can explain command-source changes on
the double-bottleneck slice: the diagnostic exposes multiple route hypotheses and the topology
command wins safety scoring on 33 steps. It does not support mitigation: the topology-guided row did
not reach success within h160, did not improve final goal distance, and did not switch between
alternate hypotheses.

Confidence is about 0.8 for this slice-local classification because the paired diagnostics used the
same scenario, seed, horizon, and step summarizer. Confidence is much lower for broader constrained
scenes because this is one bottleneck scenario and one seed.

## Instrumentation

`scripts/validation/run_topology_hypothesis_diagnostics.py` now reports
`hypothesis_switch_count` from existing per-step `topology_command_influence` rows. This fills the
issue-requested trace field without changing planner behavior.

## Evidence

- Compact evidence:
  [evidence/issue_2223_topology_hypothesis_2026-06-04/summary.json](evidence/issue_2223_topology_hypothesis_2026-06-04/summary.json)
- Evidence manifest:
  [evidence/issue_2223_topology_hypothesis_2026-06-04/manifest.md](evidence/issue_2223_topology_hypothesis_2026-06-04/manifest.md)
- Prior diagnostic context:
  [issue_1692_topology_hypothesis_probe.md](issue_1692_topology_hypothesis_probe.md)
- Candidate registry:
  [policy_search/candidate_registry.yaml](policy_search/candidate_registry.yaml)

## Implication

The current topology-guided mechanism is useful as a trace and explanation probe, but this run does
not justify promoting it as a mitigation planner. The next useful research direction is either a
mechanism change that can select non-primary hypotheses when they are safer, or a pre-registered
multi-scenario diagnostic that explicitly tests whether topology availability predicts failure
regions without claiming mitigation.

## Validation

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/validation/test_run_topology_hypothesis_diagnostics.py
```

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/validation/test_run_policy_search_candidate.py -k topology_guided
```

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_topology_hypothesis_diagnostics.py --candidate topology_guided_hybrid_rule_v0 --stage full_matrix --scenario-name classic_realworld_double_bottleneck_high --seed 111 --horizon 160 --max-hypotheses 3 --min-hypotheses 2
```

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_policy_search_step_diagnostics.py --candidate hybrid_rule_v3_fast_progress --stage full_matrix --scenario-name classic_realworld_double_bottleneck_high --seed 111 --horizon 160
```

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_policy_search_step_diagnostics.py --candidate topology_guided_hybrid_rule_v0 --stage full_matrix --scenario-name classic_realworld_double_bottleneck_high --seed 111 --horizon 160
```
