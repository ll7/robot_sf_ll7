# Guard with crossing pedestrians: robot outcome and pedestrian passage

**Diagnostic only, development seeds 1121–1130 and 1161–1200.** This is a new
current-version execution of the 0.0.8 release code plus the rebuilt diagnostic
guard switch. It is not release evidence. It changes no thesis text, claim,
release or frozen artifact. The contract ([`contract.yaml`](contract.yaml)) was
committed before any main-run episode ran.

**Question.** Does the hard-guard bundle of the continuous hybrid v4 controller
still raise collision-free completion when pedestrians must pass the same
restriction as the robot? And does it delay or block those pedestrians?

## Design

- Configurations: `G1E1` is hybrid v4 continuous with the guard on and escape on
  (its effective config equals the frozen release controller). `G0E1` is the same
  controller with the complete hard-guard bundle disabled and escape on. `orca`
  is the release ORCA config. `none` is a no-robot control: same scenario and
  seed, with the robot removed from the scene right after reset.
- Cells: ordinary doorway medium/high and bottleneck medium/high.
- Route conditions:
  - **traverse** is the release cell exactly as authored. Its pedestrian routes
    already lead through the restriction, against the robot's direction.
  - **outside** keeps the same map, robot start/goal sampling, pedestrian count,
    speed model and start states, but uses routes that never enter the
    restriction ([`inputs/`](inputs/), built by
    [`tools/build_inputs.py`](tools/build_inputs.py)).
- Size: 3 × 4 × 2 × 50 = 1,200 robot episodes, plus 400 no-robot runs. The authored
  budget is 500 steps at dt 0.1 s.
- Observation window: pedestrians are observed for a fixed 50 s window in every
  run.
- Parking rule after the robot terminates: a robot that succeeded leaves the
  scene. Any other robot stops in place and stays present.
- Pedestrian measures: passages through a gate segment inside the restriction,
  in the designed direction. These are compared with the paired no-robot run by
  pedestrian index:
  - share of no-robot passers blocked within the window;
  - mean delay to first passage, censored at the window end (a lower bound);
  - throughput loss.
- Statistics: 95% seed-block bootstrap (10,000 draws, `default_rng(1001)`), with
  equal weight per cell.
- Qualification: the pilot (seeds 1001–1010, never pooled) passed the
  no-robot gate in all four cells.

## Result (main run; all rows mode `adapter`, 0 fallback/degraded)

| Pooled over 4 cells, traverse | G1E1 guard on | G0E1 guard off | ORCA |
| --- | ---: | ---: | ---: |
| Collision-free completion, % | 100.0 | 45.0 [40.5, 49.5] | 90.0 [86.0, 93.5] |
| Wall contact, % | 0.0 | 48.5 [45.0, 52.0] | 1.0 [0.0, 2.5] |
| Pedestrian contact, % | 0.0 | 6.5 [3.5, 10.0] | 7.0 [4.0, 10.0] |
| No-robot passers blocked in window, % | 18.1 [15.6, 20.6] | 36.4 [31.5, 41.2] | 17.4 [14.8, 20.0] |
| Mean censored passage delay, s | 7.6 [6.9, 8.3] | 9.8 [8.8, 10.8] | 5.8 [5.3, 6.2] |

- Guard effect on completion (G1E1 − G0E1), traverse: **+55.0 pp [50.5, 59.5]**.
- Guard effect on completion, outside: +21.0 pp [14.0, 28.0].
- Interaction (traverse minus outside): +34.0 pp [26.0, 42.5]. The guard matters
  more, not less, when pedestrians use the passage.
- Per cell, traverse: doorway medium +88, doorway high +94, bottleneck medium +14,
  bottleneck high +24 pp. In bottleneck outside the arms do not differ (0 pp).
- Pedestrian cost of the guard (G1E1 − G0E1, traverse): delay −2.2 s
  [−2.9, −1.5]; blocked share −18.3 pp [−22.9, −13.6].

The predeclared rule gives `O1_benefit_survives_pedestrian_progress_preserved`.
"Preserved" is defined **relative to the guard-off arm**. It does not mean "no
pedestrian cost":

- With the guarded robot, pedestrians who pass without a robot are delayed by at
  least 7.6 s on average.
- 18% of them do not pass within 50 s.
- In both bottleneck cells, the guarded hybrid delays pedestrians more than ORCA
  does (+6.3 s and +3.9 s).

**Post-hoc sensitivity (not preregistered).** The rule above assumes that a
crashed robot stops in place. If a crashed robot is instead removed from the
scene, the guard's pedestrian-cost contrast changes sign: delay +2.3 s
[1.5, 3.1] and blocked share +12.5 pp [9.8, 15.1]. That would trigger O2. Robot
rows are bit-identical across the two runs (400/400). The guard's own pedestrian
cost is therefore **not identified** by this design. Two results hold under
both rules:

- the robot-outcome effect;
- the guarded arm's absolute pedestrian cost.

Details: [`evidence/sensitivity_remove_on_failure/`](evidence/sensitivity_remove_on_failure/).

Without a robot, only about half of the doorway pedestrians pass within 50 s
(366 of 750); some jam before the door. The no-robot pairing absorbs this
property of the fixture. It was not repaired.

Full tables: [`evidence/rates.csv`](evidence/rates.csv),
[`evidence/paired_effects.csv`](evidence/paired_effects.csv),
[`evidence/summary.json`](evidence/summary.json). Episode rows:
[`evidence/episodes.csv.gz`](evidence/episodes.csv.gz).

## What the design cannot separate

- **Guard-off pedestrian costs and the parking rule.** Part of the guard-off arm's
  higher pedestrian cost comes from its crashed robot stopping in the doorway,
  as the parking rule requires. See the post-hoc sensitivity in
  [`evidence/sensitivity_remove_on_failure/`](evidence/sensitivity_remove_on_failure/).
- **Scope of the estimate.** It is the effect of the whole hard-guard bundle with
  escape on, for four authored cells. It does not identify individual
  components, and it does not cover unseen layouts.
- **Pedestrian model.** The crowd model and its door jams are part of the fixture.

## Reproduce

Analysis from the compact rows (stdlib + NumPy, no simulation):

```bash
python docs/context/guard_crossing_pedestrians/analyze.py analyze \
  --episodes docs/context/guard_crossing_pedestrians/evidence/episodes.csv.gz --out /tmp/xped-check
sha256sum -c docs/context/guard_crossing_pedestrians/SHA256SUMS
```

Full acquisition (Python 3.13.14, `uv.lock` sha256 `def82098…`, CPU only), run
from this branch's contract commit:

```bash
docs/context/guard_crossing_pedestrians/tools/run_campaign.sh .venv/bin/python <out> \
  1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,$(seq -s, 1161 1200) 20
python docs/context/guard_crossing_pedestrians/analyze.py project --raw <out> --out <dir>
```

- Raw outputs (logs, per-episode JSONL, trajectory sidecars) are preserved
  outside Git: one W&B artifact plus a private host archive with a SHA-256
  manifest ([`provenance/preservation.json`](provenance/preservation.json)).
- Runtime change: `ablation_disable_hard_guard`, rebuilt byte-identically from
  the guard/escape ablation's acquisition patch. See
  [`contract.yaml`](contract.yaml) for the blob ids.
