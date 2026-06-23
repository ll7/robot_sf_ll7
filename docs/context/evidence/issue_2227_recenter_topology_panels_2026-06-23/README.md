# Issue #2227 — Static-Recenter and Topology-Guided Recovery Mechanism Panels

Contrastive mechanism panels for the two remaining #2227 sub-targets (the AMV/AMMV
sub-target was delivered in a sibling PR). For each mechanism the planner is run
twice on one fixed scenario with an identical seed, toggling **only** the mechanism
flag. The "on" arm uses a scenario where the mechanism is **expected to act**.

- **Claim boundary:** `diagnostic_only`
- **Evidence tier:** `stress` (planner-level activation accounting)
- **paper_grade:** `false`
- **Generated:** 2026-06-23
- **Commit (HEAD at run):** `fedee58afe8c5834f4e6ccb1179cdc00a8354606`

## Command

```bash
uv run python scripts/analysis/build_recenter_topology_panels_issue_2227.py
```

Raw schema-valid `simulation_trace_export.v1` traces are kept in the git-ignored
`output/issue_2227_recenter_topology/traces/` (not promoted; they are large). This
bundle promotes the rendered panels (PNG+PDF), per-mechanism captions, the per-arm
selection CSV, the run summary, and a compact trace summary.

## Mechanism 1 — static-recentering (`static_recenter_enabled`)

- **Scenario / seed:** `classic_bottleneck_low` (seed 113, horizon 160, dt 0.1) —
  the Issue #2592 static-deadlock activation-capable active row.
- **Expected to act here?** YES — static deadlock / local-minimum bottleneck where
  the recenter probe is expected to perturb the robot off the wall.
- **Activated?** YES — activation diagnostic
  `static_recenter_term_positive_in_decision_trace`: 4 recenter-term activations,
  first at step 7.
- **Command/source changed?** YES — the ON arm shifts source mix toward
  `path_follow_0.5m` (16 vs 1) as it escapes.
- **Outcome changed?** YES — ON arm reaches the goal in 122 steps (`success`); OFF
  arm exhausts the horizon (`max_steps`, `failure`). Final-pose delta 19.6 m.
- **Classification:** `activated_outcome_changed` (a real, non-fabricated delta).

## Mechanism 2 — topology-guided recovery (`topology_command_enabled`)

- **Scenario / seed:** `classic_bottleneck_medium` (seed 111, horizon 160, dt 0.1) —
  a bottleneck route-ambiguity hard slice from the topology reselection diagnostics
  (#2742 / #2716).
- **Expected to act here?** YES — bottleneck route-ambiguity slice where ≥2 distinct
  masked-route hypotheses are expected, allowing a topology-hypothesis command to be
  selected.
- **Activated?** YES — activation diagnostic
  `topology_status_counts_and_topology_hypothesis_source`: topology status `ok` on
  143 steps; the `topology_hypothesis` command source was selected 37 times.
- **Command/source changed?** YES — ON arm introduces the `topology_hypothesis`
  source (37 steps) absent from the OFF arm.
- **Outcome changed?** YES — but **not favourably**: the ON arm exhausts the horizon
  (`max_steps`, `failure`, 0 near-misses) while the OFF arm reaches the goal in 157
  steps (`success`, 9 near-misses). Final-pose delta 1.8 m.
- **Classification:** `activated_outcome_changed`.

### Honest interpretation of the topology result

This is an honest contrastive result, **not** evidence that the topology command
helps. On this slice the topology-hypothesis injection is exercised (genuine
activation) but does not improve — and in this single row degrades — the terminal
outcome relative to disabling it. This is consistent with the prior #2752 mechanism
diagnosis ("no useful topology alternative" on hard bottleneck slices). It is a
single-row, diagnostic-only observation: NOT a planner ranking, regression, or
benchmark/paper claim. The OFF-arm success here came with 9 near-misses, so neither
arm is "safe"; the panel documents mechanism behaviour, not navigation quality.

## Files

- `panels/<mechanism>/*.png`, `*.pdf` — contrastive trajectory panels (off + on arm).
- `panels/<mechanism>/mechanism_caption.md` — full per-mechanism contrastive caption.
- `representative_episode_selection.csv` — per-arm selection rows.
- `summary.json` — machine-readable run summary (activation diagnostics, deltas).
- `compact_trace_summary.json` — per-arm frame counts, activation step counts, poses.

## Reproduction notes

- Traces come from actual planner runs via `_run_map_episode(...)` with
  `record_planner_decision_trace=True` and `record_simulation_step_trace=True`.
- The two arms of a mechanism differ in exactly one config key (the mechanism flag);
  isolation is enforced and unit-tested (`test_isolation_only_flag_differs`).
- Determinism for a fixed seed/horizon is unit-tested
  (`test_static_recenter_determinism`).
- The honest-null path (mechanism expected but inactive, or active with no observable
  change) is implemented and unit-tested (`test_honest_null_path_is_representable`);
  in this run both mechanisms activated, so the null branch is exercised by tests
  rather than by the promoted panels.
