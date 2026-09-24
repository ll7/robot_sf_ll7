# Issue #9645 bounded falsification pilot

**Evidence boundary:** diagnostic local nominal execution over generated stress scenarios. This is not paper-facing benchmark evidence, a planner ranking, or real-world safety evidence. A finite search with no new counterexample does not establish that none exists.

## Decision

**NO-GO for scaling #9648 under this exact search-space and fixed-environment design.** The 64 planned evaluations completed, but all candidate tasks were easy successes, all four best-so-far curves stayed at 0.0, and no collision, timeout, route failure, or near miss appeared. Repeating the same design at larger scale is unlikely to add useful information.

This NO-GO is specific to the current fixed-seed domain. It does not say the falsification system cannot find counterexamples: the tracked #1501 archive already contains 15 collision failures under a different search contract. One archived goal-planner case was regenerated from its tracked parameters and replayed twice under current source; both current replays produced the same pedestrian collision. That case has a static `hard_but_solvable` certificate, while dynamic task feasibility remains unknown because no successful reference-planner execution was tested.

## Pilot results

| Sampler | Sampler seed | Evaluations | Valid / invalid / failed | Best score | Collisions | Near misses | Minimum clearance range (m) | Episode timestamp span (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| optuna | 1101 | 16 | 16 / 0 / 0 | 0.0 | 0 | 0 | 9.718–11.169 | 6.576 (partial) |
| random | 1101 | 16 | 16 / 0 / 0 | 0.0 | 0 | 0 | 9.809–10.746 | 12.091 (partial) |
| optuna | 2202 | 16 | 16 / 0 / 0 | 0.0 | 0 | 0 | 9.831–11.052 | 6.586 (partial) |
| random | 2202 | 16 | 16 / 0 / 0 | 0.0 | 0 | 0 | 9.814–11.048 | 6.569 (partial) |

The overall command took 39.10 seconds wall time. Per-run episode spans are partial intervals from the first episode start to the last episode end, excluding search setup and certification overhead; the run manifests do not record exact per-run driver durations. There were 64 unique candidate specs and 64 unique effective-scenario hashes.


## Convergence summary

The #9646 report was generated from the same comparison index and four persisted search manifests; report generation did not run a simulator. Each run records 16/16 native, available, scored evaluations with no invalid, failed, scoreless, missing, critical, or duplicate rows. At every evaluation point the best-so-far objective remained 0.0. The two matched sampler-seed pairs have descriptive TPE-minus-Random deltas of 0.0; no inferential test was performed. Search-level duration is absent from the run manifests; the separate 39.10-second shell wall time is recorded in `run_metadata.json`.

See [the #9646 convergence report](convergence_report.md) and [its static figure](convergence_constraints_first_lexicographic_v1.png).

Replay records were made portable by normalizing only worktree-local route, schema, and command paths. The original file hashes, normalized hashes, and rewritten fields are recorded in `path_normalization.json`; all non-path record fields were compared with the captured source outputs.

## Replays

- Pilot representative `seed_1101/optuna/candidate_0010` was a success with 11.118 m minimum human distance and 0 near misses. Its re-run matched the original status, termination, step count, seed, outcome, and selected metrics exactly. This verifies a persisted successful-case replay, not a discovered failure.
- Historical `issue_1501/failure_0002` was materialized with the existing `write_candidate_inputs` helper from the tracked #1501 archive. The current `goal` planner collided with a pedestrian after 10 steps; the event produced 1 pedestrian collision, 5 near misses, and 1.384 m minimum human distance. Two current-code replays had identical comparison signatures. The case has current `scenario_cert.v1` classification `hard_but_solvable` / `eligible`.

## Why the current search missed the known case

The pilot fixed the simulator scenario seed at 123 and clipped `start_x` to a minimum of 2.5 m after its first certification smoke landed in an inflated wall cell. The replayed historical case uses scenario seed 320 and start_x 2.312 m, so it falls outside the clipped pilot domain. The one-candidate invalid smoke was rejected before planner evaluation and was retained as a separate diagnostic attempt; the corrected one-candidate smoke completed successfully.

The current fixed-seed pilot’s minimum human distance was over 11 m and no near misses appeared. Together with the tied zero scores, this shows that the tested domain did not expose meaningful variation in planner criticality. The #1501 archive used a different objective and varied candidate scenario seeds; those results are context, not a matched comparison.

## Limits and next action

Only the `goal` baseline was evaluated. There was no planner optimization, held-out family, regression-corpus admission, alternate-planner feasibility run, or co-evolution round. Fourteen other archived #1501 failures were not replayed in this work. No candidate is reported as infeasible merely because this planner collided.

The run command included an execution-context label naming Python 3.12.3. The runner’s machine-generated provenance records Python 3.13.14 and is authoritative; the label was inaccurate and did not control the interpreter.

**Next action:** do not launch #9648. Record NO-GO and close it with this evidence after #9645’s receipt is attached. Add the replay-verified historical case to the counterexample-corpus work, then define a separate bounded pilot that includes varied environment seeds and does not exclude known admissible cases. Re-review the gate before any larger campaign.

## Machine-readable packet

- `summary.json` — design, budgets, outcomes, and decision.
- `candidate_evaluations.csv` — all 64 pilot candidates with certification, status, metrics, and checksums.
- `best_so_far.csv` — all 16 evaluation points per run.
- `smoke_attempts.csv` — the initial certification rejection and corrected successful smoke, kept outside the 64-candidate budget.
- `run_metadata.json` — exact command, source/config hashes, machine identity, and runtime notes.
- `row_status.json` — fail-closed classification for all 64 candidate executions.
- `replay_validation.json` — exact pilot-success replay and two identical current replays of one historical collision.
- `evidence_synthesis.json` — observed evidence, bounded inference, caveats, and evidence table.
- `source_hashes.sha256` — hashes for the tracked source/config/runtime inputs.
- `convergence_report.json` / `.md` and `convergence_constraints_first_lexicographic_v1.png` — the #9646 machine report, table, and convergence figure.
- `report_provenance.json` — exact convergence-report command, input digest, generator commit, and output checksums.
- `source_manifests/` — all four original 16-evaluation run manifests, retained with the comparison index and per-row checksums.
