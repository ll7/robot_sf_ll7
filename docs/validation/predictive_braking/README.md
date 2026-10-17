# Predictive stopping bound for 0.1.0

Diagnostic-only development evidence for [issue #10111](https://github.com/ll7/robot_sf_ll7/issues/10111).
No release admission, sealed-seed execution, unconditional human-yielding guarantee,
or robot-radius change is made here. Fallback or degraded execution is excluded by
the runner rather than counted as success.

## Opt-in contract

Set `v4_predictive_braking_enabled: true` on an existing
`hybrid_rule_v4_clearance_braking` configuration. The default is false, so existing
0.0.8 mappings retain their speed bands and present-position braking cap.
`v4_prediction_speed_error` is a finite nonnegative velocity-error allowance in
m/s (diagnostic value: 0.2). Enabling the switch with the braking check disabled
or on another planner variant raises an error. Binding it to a different drive
model also fails closed; the proof applies to differential-drive odometry.

The switch replaces the direction-independent present-position speed cap and
its clearance bands with per-candidate stopping certification. It preserves
candidate generation, scoring, static predicates, constant-velocity rollout
collision checks, drive limits, robot radius and all source configurations.
The retained social proximity penalties and near-human turn gate still apply.
Observed pedestrian motion can earn credit; unobserved yielding cannot.

The robot commits the candidate for at least one observed timestep (or the
configured reaction interval rounded upward), then commands zero linear and
angular speed until its actual drive limits bring it to a full stop. Forward
and reverse braking use their respective drive authorities. Midpoint/trapezoid
odometry matches the differential-drive plant.

For every stopping interval, the check minimizes relative chord separation
analytically, including the interval interior. It subtracts conservative robot
turn/interpolation padding and an error disk at the interval's end. The accepted
lower bound is strictly greater than the physical radius sum plus the larger
of the hard and braking margins. The triangle inequality proves separation
through stopping **if** the pedestrian remains within
`|p(t) - (p(0) + v(0)t)| <= v4_prediction_speed_error * t` and the observed drive
state/limits hold. This does not guarantee that a stopped robot cannot later be
struck, that protective stops in an already unsafe state have a certificate, or
that the simulator's pedestrian reactivity satisfies this bound. Measured
one-step tube violations are reported separately from actual contacts.

## Reproduction

Use the project's environment and set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
and `OPENBLAS_NUM_THREADS=1`. Run at most four workers in total.

```bash
uv run pytest tests/planner/test_predictive_braking.py tests/planner/test_hybrid_rule_v4_clearance_braking.py -q --no-cov
uv run python scripts/validation/run_predictive_braking_diagnostics.py --scenarios classic_station_platform_medium --seeds $(seq 1001 1030) --horizon 600 --workers 2 --output output/braking_station_before
uv run python scripts/validation/run_predictive_braking_diagnostics.py --predictive --scenarios classic_station_platform_medium --seeds $(seq 1001 1030) --horizon 600 --workers 2 --output output/braking_station_after
uv run python scripts/validation/run_predictive_braking_diagnostics.py --empty --workers 2 --output output/braking_empty_before
uv run python scripts/validation/run_predictive_braking_diagnostics.py --predictive --empty --workers 2 --output output/braking_empty_after
```

For linked worktrees prefix each command with
`scripts/dev/run_worktree_shared_venv.sh --`. For the fresh-main comparator, copy
this runner into the recorded base checkout and use its initialized environment.
The baseline runner initially lacked the additional audit columns; these do not
change execution. Its exact original bytes and hash are preserved with private
raw evidence, while policy-config hashes bind each public episode row.

The diagnostic runner reuses the canonical environment, policy construction,
action conversion, scenario loader, candidate/scenario overrides and pedestrian
removal hook. Its shorter loop omits the existing feasibility runner's expensive
unsampled-candidate/replay probes, which do not measure this stopping-bound
comparison. The paired arms differ only in the predictive flag. The station
budget is the same 600 steps used in the issue investigation; the empty-world
sweep retains every scenario's authored budget and covers all 48 main-matrix
scenarios on seeds 1001 and 1002 with the affected hybrid planner. No other
roster arm or doorway-width slice is included.

## Test value

All tests call the production config builder and stopping predicates; no
production test seam is added. Existing v4 tests exercise radial bands and
sampled braking, but miss retreat credit, interval-interior crossings and
bounded prediction error.

| Test | Defect / credible regression | Fails on base? | Behaviour and cheapest proof? |
| --- | --- | --- | --- |
| observed retreat | Present-position cap throttles a certifiably clear direction | Yes: 0.15 instead of 2.0 m/s | Yes; one pedestrian/candidate avoids an episode |
| crossing between samples | Endpoint-only prediction misses contact during an interval | Yes: no rejection | Yes; one 2 s interval and a 2 m/s pedestrian |
| prediction error tube | Retreat is trusted outside the configured uncertainty bound | Yes: no rejection | Yes; one candidate with a larger error allowance |
| invalid opt-in (five cases) | Disabling the sole stopping gate, wrong variant, or invalid uncertainty | Yes: no ValueError | Admission behaviour; construction is the cheapest proof |
| different drive model | Applying a differential-drive certificate to bicycle odometry | Yes: bicycle binding is accepted | Admission behaviour; one typed environment binding |
| independent odometry sweep | Accepted commands exceed the separation bound under reaction delay/error | Yes: 1.497 m at t=0.26 s, below 1.5 m | Behaviour; 64 deterministic states and the real drive, cheaper than native episodes |
| stationary pedestrian | Invented yielding or a truncated stopping horizon | No; compatibility control | Behaviour; one candidate |
| disabled switch | Silent change to an existing configuration's speed band | No; compatibility control | Behaviour; one mapping/state |

The final test file yields ten failing cases and two passing compatibility
controls on fresh main. The independent oracle uses actual drive odometry and
worst allowed inward pedestrian drift rather than computing expected values
with the helper under test. It checks straight stops; the turning allowance is
an analytic conservative bound, not empirical certification of every turn.

## Results

The compact episode tables, summary and byte proof alongside this document
record completed paired results and failure classifications.
Goal distance is distance to the current navigator waypoint, not remaining
route length; route-complete success is the primary progress measure. Minimum
centre separation and near-miss exposure use executed step endpoints, rather
than claiming continuous-time simulator contact detection. These are
exploratory dev diagnostics, correlated across shared seeds/scenarios, and
cannot establish a population-wide safety guarantee. Wilson intervals describe
episode proportions, not independent planner ranking confidence.


| Station, seeds 1001–1030 | Present-position bound | Predictive tube |
| --- | ---: | ---: |
| Route-complete successes | 0/30 | 15/30 |
| Success proportion, Wilson 95% interval | 0–11.35% | 33.15–66.85% |
| Episodes with observed contact | 0/30 | 0/30 |
| Contact proportion, Wilson 95% interval | 0–11.35% | 0–11.35% |
| Minimum executed centre separation | 1.596505 m | 1.617594 m |
| Near-miss event onsets | 102 | 200 |
| Near-miss exposure | 233.5 s | 330.1 s |
| Total exposure duration | 1800.0 s | 1767.4 s |
| Mean distance to current waypoint at termination | 10.4962 m | 4.5165 m |

Near-miss exposure is time with surface clearance in [0, 0.50) m; events count
false-to-true transitions per episode, rather than one event per exposed step.
The radius sum is 1.40 m. The improvement in progress comes with 96.1% more
near-miss onsets and 41.4% more near-miss exposure. A minimum separation statistic
and zero observed collisions do not establish unchanged social safety.

There were 834 one-step tube violations among 477,198 pedestrian transitions
(0.1748%). The largest velocity-equivalent position prediction error was
755.10 m/s: this audit includes actor discontinuities such as respawn and is
**not** a measurement of physical pedestrian speed. It is deliberately global;
it neither classifies these as nearby braking hazards nor proves long-horizon
tube coverage. The 0.2 m/s allowance therefore remains an experimental model
assumption. Accepting this opt-in experiment, calibrating a simulator-backed
bound and deciding an acceptable near-miss budget remain author decisions under
#10111. No default activation is proposed.

The paired empty-world sweep has 84/96 successes, 12/96 timeouts and zero
collisions in each arm. Every executed field compared (outcome, steps, duration,
waypoint distances, travel, separation, near-miss events/exposure and collision
steps) matches exactly. These are inherited actor-free timeouts; this braking
experiment does not establish physical infeasibility or their root cause.

| Empty-world timeout scenario | Seeds | Classification |
| --- | --- | --- |
| francis2023_frontal_approach | 1002 | Baseline actor-free timeout; unchanged |
| francis2023_pedestrian_obstruction | 1002 | Baseline actor-free timeout; unchanged |
| francis2023_pedestrian_overtaking | 1002 | Baseline actor-free timeout; unchanged |
| francis2023_robot_overtaking | 1002 | Baseline actor-free timeout; unchanged |
| francis2023_down_path | 1002 | Baseline actor-free timeout; unchanged |
| francis2023_narrow_doorway | 1001, 1002 | Baseline actor-free timeout; unchanged |
| francis2023_entering_room | 1002 | Baseline actor-free timeout; unchanged |
| francis2023_following_human | 1002 | Baseline actor-free timeout; unchanged |
| francis2023_leading_human | 1002 | Baseline actor-free timeout; unchanged |
| francis2023_accompanying_peer | 1002 | Baseline actor-free timeout; unchanged |
| francis2023_parallel_traffic | 1002 | Baseline actor-free timeout; unchanged |

The 15 remaining station failures are route-completion timeouts at 600 steps;
there are no observed-contact or execution/fallback failures. Each seed is
classified in the episode tables. No empty-world regression was introduced.

[Episode summary](summary.json), [provenance](provenance.json) and
[byte proof](release_bytes.json) are preserved beside the four per-episode CSV
files. CSV/JSON review markers deliberately remain `AI-GENERATED NEEDS-REVIEW`.
Raw logs and the original baseline runner are preserved privately; the compact
public episode rows and manifests are the durable review surface.

The paired sweeps ran at source `77669076f261c1e4c9dae94b913f3f1397d745cb`.
The subsequent change only added differential-drive admission validation:
[source comparison and native confirmation](source_equivalence.json) show
unchanged stopping/scoring methods and a seed-1003 native replay at the admission
head that matches every swept episode field exactly. Final focused checks:
54 passed; fresh-main counterexamples: 10 failed and 2 compatibility controls
passed. The first focused run before the drive restriction had 53 passes.

All config/map bytes match the starting main tree. Of the 1359 config/map files
at freeze `66f402ba176b13e45210d0da0b2cf20fcdc0cc02`, five already differed on main
before this work (listed in the byte proof); this branch changes none of them.
The frozen scenarios, planner configs, radius and release branch remain untouched.
