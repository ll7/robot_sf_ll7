# Issue #1963 Adversarial Parent Closeout

Issue: [#1963](https://github.com/ll7/robot_sf_ll7/issues/1963)
Parent: [#1488](https://github.com/ll7/robot_sf_ll7/issues/1488)

## Closeout Decision

Classification: `close_parent`

Issue #1488's bounded adversarial-search child chain has enough tracked evidence to close as a
development-stress methodology lane. It should not spawn another broad campaign by default. A
future paper-facing expansion can open a new child when there is a concrete need to repair the
crossing/TTC Optuna/TPE search space, add a new replay-confirmed failure mechanism, or scale seed
coverage.

This closeout does not upgrade any adversarial output into nominal benchmark or paper-facing
evidence. Invalid candidates, failed route trials, fallback/degraded rows, and not-available rows
remain caveats or exclusions.

## Evidence Trail

| Child | Role | Evidence | Status | Boundary |
| --- | --- | --- | --- | --- |
| Issue #1500 | Frozen manifest | `configs/adversarial/issue_1500_adversarial_comparison_manifest.v1.yaml`, `docs/context/evidence/issue_1500_adversarial_manifest/` | Complete | Specification artifact only; no execution evidence. |
| Issue #1501 | One-family crossing/TTC smoke | `docs/context/issue_1501_adversarial_smoke_run.md`, `docs/context/evidence/issue_1501_adversarial_smoke_2026-05-28/` | Complete | Smoke evidence only; replay commands existed but no repeat sweep. |
| Issue #1502 | Two-family bounded comparison | `docs/context/issue_1502_adversarial_two_family_run.md`, `docs/context/evidence/issue_1502_adversarial_two_family_2026-05-31/` | Complete | Development-stress evidence; direct cross-family failure-count comparison is invalid. |
| Issue #1503 | Stress-coverage synthesis | `docs/context/issue_1503_adversarial_stress_synthesis.md` | Complete | Synthesis over compact Issue #1502 evidence only; no new search execution. |
| Issue #1861 | Crossing/TTC replay determinism | `docs/context/issue_1861_adversarial_replay_determinism_gate.md`, `docs/context/evidence/issue_1861_adversarial_replay_2026-05-31/replay_determinism_summary.json` | Complete | Representative replay gate only; `orca` remains adapter-caveated. |
| Issue #1878 | Head-on route replay determinism | `configs/scenarios/sets/issue_1878_head_on_replay.yaml`, `docs/context/issue_1878_head_on_route_replay_determinism.md`, `docs/context/evidence/issue_1878_head_on_replay_2026-05-31/head_on_replay_determinism_summary.json` | Complete | Fixed-row determinism gate; one seed, not coverage across seeds. |

## What The Lane Proved

The lane proved that Robot SF can run a bounded, seeded adversarial comparison with explicit row
accounting and compact tracked evidence. The #1502 run completed the intended available rows:
crossing/TTC random and Optuna/TPE candidate search for `goal` and `orca`, plus guided route search
for `classic_head_on_corridor`.

The strongest observed mechanism-level findings remain:

- Crossing/TTC random search found more failures per attempted candidate than Optuna/TPE at the
  fixed #1502 budget.
- Optuna/TPE produced high failure rates among valid candidates, but its invalid-candidate rate was
  high enough that it underperformed random on failure discovery per attempt.
- Guided route search produced feasible head-on route stress trials, but its route-level objective
  is not directly comparable to crossing/TTC candidate-search failure counts.
- Representative crossing/TTC failures replayed deterministically in #1861, with `goal` native and
  `orca` through the documented adapter path.
- The selected head-on guided route now has a tracked fixture and deterministic two-pass replay in
  Issue #1878.

## What It Did Not Prove

This lane does not prove broad adversarial coverage, nominal planner ranking changes, or
paper-facing safety claims. The archive diversity is still narrow: the main archived crossing/TTC
failures are collision clusters under one template, split by policy row. The replay gates are fixed
representative checks, not seed-coverage sweeps.

Unsupported as parent closeout claims:

- direct absolute comparison between crossing/TTC failures and head-on route-search trials;
- broad mechanism diversity beyond replay-confirmed collision/head-on stress rows;
- Optuna/TPE as a useful sampler without a search-space or constraint repair pass;
- any interpretation where `invalid_candidate`, `failed_trial`, `simulation_error`,
  `fallback`, `degraded`, or `not_available` counts as success evidence.

## Parent Recommendation

Update #1488 as closeable with the `close_parent` decision. Do not reopen broad stress-search
execution from that parent. If the project needs paper-facing adversarial evidence later, open one
fresh bounded child with a narrower target, preferably one of:

- repair/constrain the crossing/TTC Optuna/TPE search space and rerun the same compact comparison;
- add one additional replay-confirmed failure mechanism or scenario family;
- expand seed coverage for the already tracked representative replay gates.

Until then, the current lane is complete as development-stress evidence with explicit caveats.

## Validation

Checked on 2026-06-01:

```bash
test -f configs/adversarial/issue_1500_adversarial_comparison_manifest.v1.yaml
test -f docs/context/issue_1501_adversarial_smoke_run.md
test -f docs/context/issue_1502_adversarial_two_family_run.md
test -f docs/context/issue_1503_adversarial_stress_synthesis.md
test -f docs/context/issue_1861_adversarial_replay_determinism_gate.md
test -f docs/context/issue_1878_head_on_route_replay_determinism.md
python -m json.tool docs/context/evidence/issue_1502_adversarial_two_family_2026-05-31/row_status_summary.json
python -m json.tool docs/context/evidence/issue_1861_adversarial_replay_2026-05-31/replay_determinism_summary.json
python -m json.tool docs/context/evidence/issue_1878_head_on_replay_2026-05-31/head_on_replay_determinism_summary.json
```
