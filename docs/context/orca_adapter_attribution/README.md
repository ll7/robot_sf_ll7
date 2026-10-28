# ORCA adapter attribution: where ORCA's request becomes zero

**Diagnostic only, development seeds 1161–1190, new current-version execution.**
This is a new execution of the 0.0.8 release code (`373dbfde4f39667cf9e8732dabe7df5118bdeab1`,
package files unchanged) on three matched-route doorway cells. It is not release
evidence and admits nothing to the thesis. 450 episodes; every row is `native`;
fallback 0 and degraded 0 are counted, not assumed.

**Question.** At 2.40 and 2.60 m, ORCA on the differential-drive robot completes no episode
although the opening is physically feasible. Is the zero request produced by the RVO2
solver, by the adapter's heading-safe projection, or between the executed command and
the measured motion? And does native holonomic execution change the outcome?

## Result

| Configuration | 2.40 m | 2.60 m | 2.80 m |
| --- | ---: | ---: | ---: |
| ORCA, differential drive | 0/30 | 0/30 | 27/30 |
| ORCA, native holonomic | 0/30 | 0/30 | 0/30 |
| Social force, differential drive | 0/30 | 0/30 | 0/30 (30 pedestrian contacts) |
| Social force, native holonomic | 0/30 | 0/30 | 0/30 (30 pedestrian contacts) |
| Hybrid v4 (continuous), differential drive | 0/30 | 24/30 | 24/30 |

* **The solver requests zero (rule R3 holds).** In all 60 of 60 failed ORCA
  differential-drive episodes at 2.40/2.60 m, the last 10 s are dominated by steps in which
  the raw RVO2 world velocity is below 0.05 m/s while the preferred velocity stays at about
  3 m/s toward the goal (A1 = 1.00, seed-block 95 % interval [1.00, 1.00]). No failed episode
  is dominated by projection suppression (A2 = 0/60). The projection does zero a goal-ward
  request transiently (8 steps in 4 episodes at 2.40 m; 118 steps in 17 episodes at 2.60 m),
  but not where the robot finally stops.
* **The request fades at the mouth of the opening.** The final zero run starts with the robot
  centre at x ≈ 14.2 m (2.40 m) and 14.4 m (2.60 m). The 1.0 m disc then reaches 0.2–0.4 m into
  the 1 m deep opening (wall x = 15–16 m), with no pedestrian nearer than 3.3 m. The median
  solver speed in the 5 s before that is 0.04 m/s (exploratory).
* **The obstacle representation leaves less room than the robot needs.** The free gap at the
  wall between the inflated obstacle squares the adapter passes to RVO2 has median 1.61 m
  (2.40 m cell) and 1.69 m (2.60 m cell) over all steps, against a 2.0 m robot diameter. This
  gap is a descriptive geometry readout, not a sharp threshold: successful 2.80 m episodes
  also saw gaps of 1.67–1.95 m.
* **Native holonomic execution does not remove the deficit and is worse at 2.80 m.** The paired
  success difference holonomic minus differential drive is 0 at 2.40 and 2.60 m (both arms 0/30,
  a shared floor) and −0.90 at 2.80 m (95 % interval [−1.00, −0.80]); holonomic ORCA also ends
  with a zero solver request there (30/30), with median terminal gap 1.83 m against 2.00 m on
  differential drive.

Rules (predeclared in [`contract.yaml`](contract.yaml)): R1 false, R2 false, R3 **true**,
R4 **true** (2.80 m only), R5 true only as a shared 0/30 floor, so it bounds no effect
beyond these cells. Stage 2 (head-on, bottleneck, crossing) was **not run**: its trigger
required neither R2 nor R3 to hold.

## What the design can and cannot separate

* The step trace separates a zero solver request from a suppressed nonzero request inside the
  executed differential-drive stack. That is the question the earlier width series could not
  answer.
* Native holonomic execution changes motion authority (instant velocity, no heading,
  acceleration or turn-rate limit) as well as removing the projection. It still uses the same
  grid-based obstacle inflation of the ORCA adapter. It therefore does not isolate a pure
  projection effect, does not test "ORCA without the adapter", and is not an algorithm ranking.
* The heading still enters the solver input through the corner-inflation rule and the robot's
  current velocity. "The solver requests zero" means the immediate zero comes from the solver
  given its inputs. It does not mean that the approach shaped by earlier projected commands
  played no part.
* The cells reconstruct the matched-route width series (base doorway map, route `[[7,5],[25.5,5]]`,
  gap widened symmetrically) because the original packet was not available. Pilot seeds
  1001–1010 qualified the reconstruction (ORCA differential drive 0/10, 0/10, 10/10; terminal x
  14.20/14.50 m), and the main seeds reproduce the reported 0/30 and 0/30 exactly. At 2.80 m
  the result is 27/30 against the reported 25/30 on other seeds.

## Packet

* [`contract.yaml`](contract.yaml): preregistration, committed in `a4a31e357` before any main
  episode ran. Same commit for all arms. The only later change to the predeclared analysis is a
  `--steps` option naming the step-table file; the computation is unchanged.
* [`run_xorca.py`](run_xorca.py): harness that wraps planner and step-loop functions at runtime and
  adds an `xorca` record (preferred velocity, raw solver velocity, occupancy penalty, adapter
  command, obstacle gap, pre/post pose) to each step of the runner's own
  `simulation_step_trace`, which already holds the projected command and environment action.
  [`provenance/neutrality.json`](provenance/neutrality.json): with and without the wrappers, 15 of 15
  episodes are identical step by step.
* [`acquire.py`](acquire.py), [`extract.py`](extract.py), [`analyze.py`](analyze.py) and
  [`explore.py`](explore.py) (exploratory, outside every rule).
* `evidence/`: `episodes.csv` (one row per episode with `mode`), `orca_steps.csv.gz` (ORCA
  step table, 6 decimals), `cell_table.csv`, `paired_effects.csv`, `summary.json`,
  `exploratory.json`.
* `provenance/`: environment identity, job receipts, pilot table, custody (W&B artifact
  `wandb://ll7/robot_sf/xorca-orca-adapter-attribution:v0` with raw episode JSONL and
  full step traces, raw-bundle checksums).

## Reproduction

The analysis needs only Python and NumPy/PyYAML. Run it from the repository root and write to a
directory outside the packet:

```bash
P=docs/context/orca_adapter_attribution
uv run python $P/analyze.py --contract $P/contract.yaml --evidence $P/evidence \
  --steps orca_steps.csv.gz --output /tmp/xorca-check
for f in summary.json cell_table.csv paired_effects.csv; do cmp $P/evidence/$f /tmp/xorca-check/$f; done
(cd $P && sha256sum -c SHA256SUMS)
```

Full acquisition (about 40 min of single-core episode time; about 6 min on 14 workers), from a
checkout of this branch with `uv sync --frozen`:

```bash
P=docs/context/orca_adapter_attribution
uv run python $P/acquire.py --contract $P/contract.yaml --stage main --out-dir <run_dir> --workers 14
uv run python $P/extract.py --raw-dir <run_dir> --stage main --contract $P/contract.yaml --out-dir <ex_dir>
uv run python $P/analyze.py --contract $P/contract.yaml --evidence <ex_dir> --output <ex_dir>
```
