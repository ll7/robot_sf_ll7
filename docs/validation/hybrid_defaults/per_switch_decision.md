# Hybrid defaults: decision after the review fixes

**The code retains the author's all-on ruling (author decision 2026-10-05).**
Reopen: **author: can still be discussed in more detail**.
The recommendation is for the author to consider.

Development evidence only: 48 standard scenarios × dev1001–1030, five executable
configurations (7,200 episodes), plus 204 empty-world episodes. Two local simulation
workers; matched scenario budgets of 400–700 steps, recorded at 0.1 s per step.
Literal goal-only was additionally checked in all 1,440 cells: missing `next_valid`
raises before any environment step. Flags below mean physical exclusion / goal
validity / validity sensor. These experiments stepped only development seeds.

| Option | Success / collision / timeout (1,440) | Success delta from base | Paired time-to-goal delta; common successes |
| --- | ---: | ---: | ---: |
| Base, all off (000) | 1,285 / 0 / 155 | — | — |
| Static alone (100) | 1,262 / 1 / 177 | −23 | +1.039 s; 1,233 |
| Sensor alone (001) | 1,285 / 0 / 155 | 0 | 0 s; 1,285 |
| Validity with its required sensor (011) | 1,342 / 0 / 98 | +57 | −0.183 s; 1,283 |
| All on, current default (111) | 1,318 / 1 / 121 | +33 | +0.679 s; 1,251 |
| Literal validity alone (010) | 1,440 configuration errors | Not an outcome denominator | No environment steps |

Each highlighted cell below is success / collision / timeout out of 30.

| Scenario | 000 | 100 | 001 | 011 | 111 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `francis2023_narrow_hallway` | 26/0/4 | 5/0/25 | 26/0/4 | 30/0/0 | 5/0/25 |
| `francis2023_robot_crowding` | 19/0/11 | 15/1/14 | 19/0/11 | 19/0/11 | 18/1/11 |
| `classic_bottleneck_high` | 30/0/0 | 29/0/1 | 30/0/0 | 30/0/0 | 29/0/1 |
| `francis2023_exiting_room` | 30/0/0 | 30/0/0 | 30/0/0 | 29/0/1 | 29/0/1 |

Both collisions are robot-crowding seed 1013, in 100 and 111: pedestrian contact at
5.8 s, zero robot displacement on contact, after 0.8 s continuously stationary
in `PROTECTIVE_STOP`, with no feasible moving candidate. They remain collisions.
All 724 outcome failures, including empty-world, are classified. No fallback
or degraded execution occurred.

The hard terminal `GOAL_STOP` now uses navigator completion and passes the 0.22 m
regression. Two new 011 failures remain relative to base: exiting room and exiting
elevator, both seed 1026. They are scorer livelocks: `NORMAL` prefers a zero route-guide
command despite 59 feasible moving candidates, rather than a premature `GOAL_STOP`.
011 recovers 59 old failures and introduces two; 111 recovers 67 and introduces
34. Sensor-only and fixed all-off controls match the baseline in every cell.

**Recommendation:** consider 011, which records 24 more successes and one fewer
collision than 111. Keep physical exclusion opt-in until its narrow-hallway loss
and contact case are addressed. If accepting no newly failing cells is required,
001 or 000 are the measured alternatives while the two scorer livelocks are repaired.
The author must weigh these consequences under the reopen clause.

Empty-world 000→111: 88→100 successes of 102, zero contacts in both, 14→2 timeouts,
+0.276 s over 88 common successes. This two-arm sweep cannot attribute the gain
to an individual switch. All 183 catalogued release arms retain full base config
dumps, 179 canonical mapping digests match, and four historical unfrozen guards
remain blocking. Released learned observation spaces stay unchanged, including
older PPO and config-less arms.

[All 48 scenarios and times](per_switch_scenarios.md) ·
[Every failure and paired deltas](per_switch_summary.json) ·
[Source and budget identity](per_switch_manifest.json) ·
[Final runtime applicability](measurement_applicability.json) ·
[Tests and byte preservation](review_fix_proof.md).
