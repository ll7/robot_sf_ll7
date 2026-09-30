# Scenario dynamic-window diagnostic

Diagnostic only. This packet tests the rationale for short authored scenario
budgets; it does not change the release contract or admit benchmark evidence.
The expanded scenario copy is frozen from the canonical 0.0.8 template loader.
The 800-step sidecar overrides every scenario; five dev seeds (1001–1005) and
three arms yield 720 episodes, at dt=0.1 s. Authored budgets remain metadata.

Use **only** `scripts/validation/run_hzev_dynamic_window.py` with this config.
The ordinary campaign CLI does not implement this diagnostic's stationary arm
or continuation. The stationary entry uses the goal observation contract and
replaces its policy with the exact command `(0, 0)`; no goal inference occurs.
Movers use the existing native goal and ORCA policy builders. ORCA requires
`rvo2`; missing dependencies fail preflight. No fallback is accepted.

The harness records reset-time robot start and **final sampled route goal**,
pedestrian positions at t=0 and each step, active behavior types, route group
counts, actual route-end respawns, and the first ordinary terminal event. It
continues the same environment after contacts, with no reset. Movers park after
first successful route completion. This gives an 80-second observation window
without letting a completed mover repeatedly traverse the route. Full-window
last-interaction times include the parked tail and post-contact continuation;
the last interaction up to the first terminal event is also exported. These
extensions are diagnostic and cannot be interpreted as benchmark scores.

The ordinary episode record ends at the **first ordinary terminal event**.
Every trajectory, event ledger, outcome flag and metric input is frozen there,
then passed through the unchanged production metric calculation and integrity
validator. `algorithm_metadata.simulation_step_trace` contains only that prefix.
The complete 800-step window lives at `hzev.diagnostic_trace.steps`; analysis
reads that separate trace. `hzev.first_collision` and `hzev.first_goal` record
the observed event order independently of the ordinary outcome. A later goal
after collision never converts the ordinary collision into success; contact
after an early success never converts the ordinary success into collision.
Both new and resumed rows validate the ordinary prefix and diagnostic geometry.

## Acquisition and recovery

From a checkout of the pushed diagnostic branch, the orchestrator submits:

```bash
sbatch scripts/validation/hzev_dynamic_window.sbatch <full-diagnostic-sha>
```

The script checks out that exact SHA in `$HOME/hzev` (override `HZEV_WORK` for
a dedicated clean clone). It uses partition `a30`, QoS `a30-cpu`, four allocated
CPUs, single-threaded math libraries, and three workers plus their parent.
Workers are `SLURM_CPUS_PER_TASK - 1`; requesting 16 CPUs gives 15 workers.
The CLI accepts 1..`SLURM_CPUS_PER_TASK` (or `os.cpu_count()` outside Slurm),
with default three. The local repair lane uses at most three simulation workers.
The default time is 55 minutes: the six-scenario H800 sample maximum was
8.738 seconds per episode; `720/3 * 8.738 * 1.5` is 52 minutes 26 seconds,
rounded up. This is a local CPU estimate with 50% margin, not measured cluster
throughput; installation and preservation also consume allocation time.
It writes outside the checkout to `$HOME/hzev_data/at_<sha>_<jobid>`; override
`HZEV_OUT` if desired. Keep the directory and generated tarball/checksum, and
copy the tarball to durable external storage before cluster access ends.

Each episode JSON is atomically replaced, and the manifest retains its checksum.
An interrupted run leaves `complete=false` or missing rows. Re-run the acquisition
command with the identical SHA/config/roster and output directory; valid complete
rows are read back and reused, while incompatible rows fail closed. `.partial`
files never count as completed episodes. Full analysis requires all 720 keys,
one source identity, valid checksums, and complete 800-step traces. A fresh source
or configuration requires a fresh output directory.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python scripts/validation/run_hzev_dynamic_window.py \
  --head-sha <sha> --workers 3 --output-dir <outside-checkout-directory> --check-only
# Remove --check-only to acquire; the orchestrator owns execution.
.venv/bin/python scripts/validation/analyze_hzev_dynamic_window.py \
  --input-dir <directory> --output <directory>/analysis.json
```

## Definitions and interpretation

`T_clear` is the first sampled time after the last occupied sample of the finite
start-to-final-goal corridor. The corridor is a capsule with half-width 1.5 m;
being within a further 3 m of it means distance to its center segment <=4.5 m.
For ambiguity checking, the per-seed output also reports the 3 m centerline
interpretation. Positions are center coordinates; sampling uncertainty is dt.
Clear from reset gives T_clear=0. Occupied at step 800 is right-censored, never
assigned a made-up T_clear=80. An observed clear tail is bounded by 80 seconds,
not mathematical proof that no pedestrian will ever return.

Recurring route groups respawn at their start; populated crowded-zone controllers
continuously retarget goals. Their `T_clear` is unavailable even if they happen
to have a clear final tail. Behavior inventory and actual respawn counts support
this classification, rather than inferring endless dynamics from one late actor.
Scripted single pedestrians are not removed on reaching their goals and may
remain near the corridor. Robot-relative follow/lead/accompany roles may differ
between stationary and moving arms.

The requested third label, “no persistent pedestrian flow (routes respawn, so
the dynamics never end)”, is retained verbatim for interoperability. Its wording
is contradictory: route respawn describes **persistent/recurring flow**. The
machine field `recurring_flow` conveys the operative meaning.

`T_last_interaction` is the final observed time a pedestrian center is <=2.5 m
from the moving robot center, reported separately for ORCA and goal. No observed
interaction is null (not zero); an interaction at the horizon is flagged.

Wait-then-go time is `T_clear + straight_distance/2.0 + 2.0 seconds` by default.
The margin is configurable and stored in analysis provenance. It is converted
to steps using the actual dt, and compared with 400, 500, 600 and the authored
budget. Median and maximum use available observations, with unavailable counts;
per-seed classifications and disagreements remain visible. Any exploitable seed
flags a scenario as potentially wait-exploitable; this is not an all-seed result.
When clearance is censored, a lower bound may suffice to show the authored budget
is inside the observed dynamic window; otherwise the result stays unresolved.

This arithmetic is an **optimistic timing test**, not proof of a viable strategy.
Static walls, bends, goal-zone entry, acceleration and pedestrian reactions to
a departing robot can invalidate straight-line completion. The matrix includes
a declared infeasible doorway probe. A passing timing test needs a separate
wait-then-go execution before claiming real exploit success. Conversely, the
short local smoke cannot classify a scenario; partial analysis is explicit.

## Validation

Synthetic controls protect finite-segment geometry, both corridor interpretations,
re-entry after a clear interval, seconds-to-steps conversion, authored budgets,
recurring flow, right censoring, no-interaction handling, seed median/max and
malformed trace rejection. A first-occupied instead of last-occupied mutation
must fail the re-entry test. Existing camera-ready tests cover horizon binding
and trace emission, but not these new dynamic-window quantities. The tests need
no production-only seam; values follow hand calculations.

The repair runs classic_bottleneck_low/1002 in all arms at 800 steps, real mover
collision-then-goal and early-goal witnesses, and six scenarios × one dev seed
× three arms at 800 steps. Controlled 800-step tests cover collision then goal,
goal then collision, simultaneous contact/goal and timeout in all three arms.
A real stationary goal witness is generally impossible under its zero-command
contract; the stationary event-order regression is explicitly controlled.
Two real short seeded episodes compare one versus three workers; all result
fields match except timestamps, wall time and measured steps per second.
Short CLI smoke permits <=100 steps, one scenario and one dev seed, with any
selected arms and allocated worker count. Full local samples still use
`--allow-partial` analysis because they do not contain the complete 720-row
roster. They prove acquisition compatibility, not full-matrix conclusions.
