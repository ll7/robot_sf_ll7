# Hybrid failures: root causes and regression proof

Author direction, 2026-10-07: fix the known failures before deciding defaults.
The all-on code default remains unchanged. These are development diagnostics,
not release admission. Baseline: `9dd6009f8499713925da65c2ecc0d4962a115031`;
implementation producer: `a635fb00e95e5446ea20e8dda49a879a899e2930`.

## Reflection

The first mirror-Y command divergence occurs at step 4. The same 1 m/s straight
candidate has different raster center clearance in the reflected grid; mirror-X
first diverges at step 33 at a cell boundary. Candidate rejection counts are
identical. The winner changes because a discontinuous raster score is used on
an otherwise continuous physical rollout. Replacing only clearance lookup by
exact center distance removes both first divergences. The fix uses continuous
world geometry whenever the physical gate is enabled. Switch-off keeps its
historical raster path. The complete fifteen-episode dev1001, 60-step gate now
passes both reflections for every arm: maximum error **0.002211 mm**, against
**0.1 mm**. Previously static and all-on reached **373.754 mm**.

## Hallway and crowding

The static switch also replaced the rollout used by the pedestrian forecast and
progress scorer, and used discontinuous raster clearance for a surface-gap comfort score.
The rollout coupling is separate from physical exclusion.
The isolated counterfactuals on the same failing cells are:

| Change alone | Hallway dev1001 | Crowding dev1013 |
| --- | --- | --- |
| Exact clearance lookup | timeout, 40 s | contact, 5.8 s |
| Remove surface-gap scoring (rejected counterfactual) | success, 18.1 s | contact, 5.8 s |
| Restore baseline rollout/scoring context | success, 22.3 s | success, 37.0 s |

Hallway first diverges at step 14: the old route-guide winner is displaced by a
candidate rewarded for raster clearance. Later it repeatedly stops or creeps;
by step 200 its pedestrian speed cap is 0.15 m/s, leaving 3.337 m at the horizon.
Crowding first diverges at step 6 even with saturated static-clearance scores.
The coupled rollout changes the approach; at step 45 all 68 candidates fail the
pedestrian forecast, and braking leaves the robot stationary from step 50.
At step 57 (5.8 s), a pedestrian contacts it. The all-on contact follows the same
trace. Exact geometry alone does not prevent it, so it is not the mirror defect.

A tighter crowding counterfactual keeps the original plant static gate, exclusion
radius and surface-gap score, changing only the forecast/scoring trajectory.
It still recovers dev1013 at 37.0 s. Removing surface-gap preference is unnecessary
and violates the existing `test_opt_in_clearance_preference_uses_surface_gap`;
that existing scoring contract is retained.

The fix evaluates swept physical-body exclusion in its own plant-accurate
rollout, including the committed interval and existing braking tail. It does
not replace the common pedestrian forecast or progress rollout. The surface-gap
comfort score remains, with continuous center-distance lookup before subtracting
the robot radius. The final real-path regression completes hallway in **26.7 s** and crowding
in **37.0 s**, with no contacts or fallback. The full matrix below is needed to
assess effects beyond these witnesses.

## Terminal scorer stalls

The route guide has both goal tolerance and a 0.3 m waypoint stopping threshold.
The previous fix cleared only goal tolerance. Clearing the waypoint threshold
produces a moving guide candidate, but alone still times out in both dev1026
cells. A second defect is in the score: after a rollout passes the terminal
target, heading alignment uses the vector from its predicted endpoint back to
the goal. It penalizes a successful approach by nearly 180 degrees and rewards
zero motion. This is not an extra source bonus; the scores come from alignment
and the other existing terms. A one-variable approach-heading probe recovers
both cells (13.5 s room, 15.0 s elevator).

Terminal tracking now clears both guide thresholds and scores approach heading
from the current pose to the terminal target. Nonterminal and disabled-validity
paths retain the old formula. Actual navigator completion still owns GOAL_STOP;
no distance-only hard stop is reintroduced. Both native regressions pass.

## Tests and preservation

Before any fix, this command fails **6/6** on baseline, for the actual trajectory,
collision or timeout assertions:

```sh
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest -n 2 tests/planner/test_hybrid_failure_regressions.py -q
```

Afterward all six pass. The combined focused command in
[root_cause_proof.json](root_cause_proof.json) passes **183 tests**. It includes
all 183 constructor/environment dumps, 179 canonical mapping comparisons and
four unchanged historical guards, all nine frozen planner snapshots, 48 scenario
environments and explicit switch overrides. All 60 registry sources and their
62 source/dependency paths retain exact baseline bytes. No frozen YAML is edited.
Released learned observation contracts retain their registered legacy defaults;
this round adds no sensor or model-space change.

| Test cases | Bug caught; credible regression | Why existing coverage misses it | Deterministic / real path / seam |
| --- | --- | --- | --- |
| `test_physical_static_rollout_reflects_without_raster_score_bias` (two reflections) | Reintroducing raster scoring on the exact physical path changes reflected commands. Fails twice on baseline. | The older diagnostic mirror witness explicitly selects all-off. | Fixed dev1001 scene, 60 real env steps; actual policy, sensor and drive. Harness configuration patch only; no production seam. |
| `test_enabled_switch_completes_the_reproduced_failure_cell` (hallway, crowding) | Coupling pedestrian/scoring rollout to the static switch, or restoring discontinuous raster scoring, restores the timeout/contact. Both fail on baseline. | Physical overlap tests do not follow the resulting crowd interaction to completion. | Exact scenario and dev seed; native producer, environment, planner and action conversion. No seam. |
| Same test (room, elevator) | Restoring either terminal waypoint stopping or endpoint heading reversal stalls despite moving candidates. Both fail on baseline. | The 0.22 m synthetic stop test lacks actual route-grid and released scorer weights. | Exact dev1026 scenarios, native sensor/guide/scorer/drive. No seam. |

[Before-fix mirror candidate scores](root_cause_candidate_scores.json.gz),
[selected complete per-step decisions](root_cause_step_traces.json.gz),
[probe source hashes](root_cause_probe_identities.json), and the
[fifteen full mirror trajectories](mirror_switch_probe.json.gz) retain the direct
witnesses. Native matrix evidence remains diagnostic-only; defaults remain the
author's decision under the existing reopen clause.

The one full-suite run at the preceding implementation passed 43,003 tests,
with four failures. One exposed a scoring-unit change; the surface-gap contract
was restored, and that unchanged test is included in the 183 passing focused
tests above. Two candidate-materialization checks rejected tracked evidence
being written during the run. The artifact walkthrough rejected a pre-existing
output directory. The prior walkthrough output was preserved separately. The
three infrastructure cases are rechecked on a clean tracked tree and fresh
walkthrough output; their results accompany the completed measurement report.

## Full-matrix qualification: hallway remains unresolved

The completed 7,404-cell measurement removes both crowding contacts and the
reported terminal scorer stalls, and passes the mirror gate. It does **not**
resolve the hallway collapse: off/static/sensor/validity/all-on successes are
26/4/26/30/5 of thirty. The dev1001 witness was insufficient to establish a
general repair. Do not read its passing regression as a hallway-wide fix.

In the remaining dev1003 static-only failure, the first command divergence is
step 22: off chooses route-guide `[2.0, 0.402112]`; static chooses dynamic-window
`[1.7999999, 0.145064]`. The changed approach subsequently enters the pedestrian
speed cap. At steps 200, 300 and 399 the cap is 0.15 m/s despite 56 feasible moving
candidates; final goal distance is 2.664846 m at 40 s. This is not a terminal
GOAL_STOP or an all-candidates-rejected static gate.

Three one-variable native counterfactuals keep the physical swept gate and all
dynamic hard constraints: center-score normalization finishes at 24.3 s; restoring
raster lookup while retaining surface-gap units times out at 40 s; removing only
the static comfort weighted contribution finishes at 24.3 s. The center-score
change would violate the existing surface-gap unit regression and is not adopted.
No weight or default change is made from these probes. [Full counterfactual
traces and intervention identity](remaining_hallway_probe.json.gz) are separate
from the immutable matrix and support the remaining repair and author decision.

The terminal state is AUTHOR_DECISION_REQUIRED; item 2 is not fully resolved and
the draft is not merge-ready. The recommendation is validity with its sensor,
with static exclusion opt-in pending further hallway repair.
