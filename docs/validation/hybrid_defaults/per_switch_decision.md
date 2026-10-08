# Hybrid defaults: author decision brief

**Author ruling, 2026-10-08 (chat to the orchestrator): choose 011, goal
validity with its sensor; keep static exclusion opt-in.** This supersedes the
October 5 all-on choice. The selected arm has 1,346/1,440 successes against
1,285 off and 1,319 all-on, recovers 61 episodes and introduces none. Static
exclusion introduces 48 failures and drops hallway success from 26/30 to 4/30.
All arms have zero collisions in this sample. Opt in with
`physical_static_exclusion_enabled: true`; the hallway defect remains open.

**Reopen if a fixed static exclusion beats validity plus sensor without introducing failures.**
[Ruling proof](ruling_proof.md), [custody manifest](preservation_manifest.json.gz),
and [selection applicability](ruling_applicability.json) accompany this decision.

## Historical packet submitted before the October 8 ruling

**All-on remains in code.** Reason: **author decision 2026-10-05**.
2026-10-07 direction: fix known failures before choosing the default.
Reopen: **author: can still be discussed in more detail**.
**AUTHOR_DECISION_REQUIRED; item 2 remains incomplete; draft not merge-ready.**

Measured the same 48 scenarios × dev1001–1030 × five arms (7,200 episodes),
204 empty-world episodes and 15 mirror episodes, with two simulation workers.
Budgets: 400–700 steps, 0.1 s. Flag order: static exclusion / goal validity / sensor.
Literal 010 fails closed; its previous missing-sensor proof is historical.

| Option | Before S / C / T | After S / C / T (1,440) | Δ successes vs off | Paired seconds vs off; common successes | Recovered / introduced vs off |
| --- | ---: | ---: | ---: | ---: | ---: |
| All off (000) | 1,285 / 0 / 155 | 1,285 / 0 / 155 | +0 | +0.000; 1,285 | 0 / 0 |
| Static exclusion (100) | 1,262 / 1 / 177 | 1,250 / 0 / 190 | -35 | +0.492; 1,237 | 13 / 48 |
| Validity sensor (001) | 1,285 / 0 / 155 | 1,285 / 0 / 155 | +0 | +0.000; 1,285 | 0 / 0 |
| Goal validity with required sensor (011) | 1,342 / 0 / 98 | 1,346 / 0 / 94 | +61 | -0.231; 1,285 | 61 / 0 |
| All on (current default) (111) | 1,318 / 1 / 121 | 1,319 / 0 / 121 | +34 | +0.190; 1,254 | 65 / 31 |

Highlighted scenario cells: **after S/C/T**, with before successes in parentheses.

| Scenario | 000 | 100 | 001 | 011 | 111 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `francis2023_narrow_hallway` | 26/0/4 (26) | 4/0/26 (5) | 26/0/4 (26) | 30/0/0 (30) | 5/0/25 (5) |
| `francis2023_robot_crowding` | 19/0/11 (19) | 19/0/11 (15) | 19/0/11 (19) | 19/0/11 (19) | 19/0/11 (18) |
| `classic_bottleneck_high` | 30/0/0 (30) | 29/0/1 (29) | 30/0/0 (30) | 30/0/0 (30) | 29/0/1 (29) |
| `francis2023_exiting_room` | 30/0/0 (30) | 30/0/0 (30) | 30/0/0 (30) | 30/0/0 (29) | 30/0/0 (29) |


Continuous clearance fixes reflection (maximum 0.002211 mm; gate 0.1 mm).
Separate swept physical and pedestrian/scoring rollouts remove both crowding
contacts. Clearing both terminal guide thresholds and scoring current approach
heading removes the reported room/elevator stalls. Six new regressions fail on
9dd6009f and pass after; the full matrix refutes a general hallway repair.

**Recommendation:** author should prefer 011: +61 successes, zero introduced
failures versus off, and −0.231 s on common successes. Keep static exclusion
opt-in pending hallway repair. This recommendation does not change the code.
Hallway dev1003 diverges at step 22; later it stays at 0.15 m/s despite 56 moving
candidates. Removing only the static comfort contribution finishes at 24.3 s;
restoring raster lookup alone does not. The surface-gap unit regression still
applies; no score-unit or weight change is adopted. [Counterfactual traces](remaining_hallway_probe.json.gz)
are outside the matrix denominator.

All 731 failures are classified: 183 forced-stop, 461 moving-candidate horizon
exhaustions, 87 low-progress/livelock timeouts; zero collisions or fallback.
Empty-world off→all-on: 88→100 successes / 102, zero collisions, 14→2 timeouts;
−0.108 s over 88 common successes. The two-arm sweep cannot isolate switches.

Released identities remain unchanged: 183 full dumps, 179 mappings plus four
historical guards, nine frozen planner and 48 environment snapshots; 62 source
files unchanged. Released learned observation spaces retain their contracts.
All 1,440 crowded and 102 empty off controls match previous outcome/time cells;
sensor-only matches off exactly. The full suite ran once: 43,003 passed, four
classified failures. One scoring-unit regression was corrected; three clean/fresh
infrastructure rechecks and 183 focused tests pass. No final-head full-suite claim.

External disk exhaustion interrupted the run. All 4,609 retained JSON/gzip pairs
were verified; 2,795 remaining cells resumed under identical producer/input
identities. All 7,404 pairs pass completion checks. No disk pauses were needed;
the owned process group had a 25 GiB pause guard. [Integrity receipt](resume_validation.json.gz).

[All 48 scenarios](per_switch_scenarios.md) · [Counts, paired times, recovered/introduced cells and classifications](per_switch_summary.json) ·
[Root causes and test value](root_cause_proof.md) · [Applicability](measurement_applicability.json) ·
[Suite qualifications](review_validation.json).
