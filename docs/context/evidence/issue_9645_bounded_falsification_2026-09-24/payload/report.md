# Issue #9645 bounded falsification pilot

**Evidence boundary:** diagnostic local nominal execution over generated stress scenarios. This is not paper-facing benchmark evidence, a planner ranking, or real-world safety evidence. A finite search with no new counterexample does not establish that none exists.

## Decision

**NO-GO for scaling #9648 under this exact search-space and fixed-environment design.** All 64 planned evaluations completed with recorded task success, zero collisions, timeouts, route failures, and near misses; minimum clearance was at least 9.718 m, and all four best-so-far curves stayed at 0.0. Severe-intrusion evidence is absent for every candidate, so safety criticality is unknown for all 64. These observations do not justify scaling this unchanged design; they do not establish that the cases were non-critical.

This NO-GO is specific to the current fixed-seed domain. It does not say the falsification system cannot find counterexamples: the tracked #1501 archive already contains 15 collision failures under a different search contract. One candidate was regenerated from archived parameters and replayed twice at recorded source revision `58e516aa4f69ff3098bf518199f483006589758c`; both replays produced the same pedestrian collision. Exact input binding to the historical execution remains unknown, so the candidate is pending and not admitted to the corpus. Its regenerated scenario has a static `hard_but_solvable` certificate, while dynamic task feasibility remains unknown because no successful reference-planner execution was tested. No replay at the current PR head is claimed.

## Pilot results

| Sampler | Sampler seed | Evaluations | Valid / invalid / failed | Best score | Collisions | Near misses | Minimum clearance range (m) | Safety criticality | Episode timestamp span (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| optuna | 1101 | 16 | 16 / 0 / 0 | 0.0 | 0 | 0 | 9.718–11.169 | unknown (16/16) | 6.576 (partial) |
| random | 1101 | 16 | 16 / 0 / 0 | 0.0 | 0 | 0 | 9.809–10.746 | unknown (16/16) | 12.091 (partial) |
| optuna | 2202 | 16 | 16 / 0 / 0 | 0.0 | 0 | 0 | 9.831–11.052 | unknown (16/16) | 6.586 (partial) |
| random | 2202 | 16 | 16 / 0 / 0 | 0.0 | 0 | 0 | 9.814–11.048 | unknown (16/16) | 6.569 (partial) |

The overall command took 39.10 seconds wall time. Per-run episode spans are partial intervals from the first episode start to the last episode end, excluding search setup and certification overhead; the run manifests do not record exact per-run driver durations. There were 64 unique candidate specs and 64 unique effective-scenario hashes.


## Convergence summary

The corrected #9646 schema-v3 report was rebuilt from the byte-bound comparison, four derived digest-pinned manifests, and all 64 exact producer episode records; report generation did not run a simulator or search. Each run records 16/16 native, available, scored evaluations with no invalid, failed, scoreless, missing, or duplicate rows. All 64 episode-record hashes match their recorded producer digests, and their embedded provenance consistently identifies source revision `58e516aa4f69ff3098bf518199f483006589758c`. Both matched Random/TPE seed pairs are eligible for descriptive comparison and tie at 0.0; no inferential test was performed. Every candidate's criticality and collision/severe-intrusion tier remains unknown because severe-intrusion evidence is absent. The stored scores remain all 0.0, but the v1 scorer can produce its negative composite when one component is absent; those historical scores do not establish a negative safety tier. Search-level duration is absent from the run manifests; the separate 39.10-second shell wall time is recorded in `run_metadata.json`.

See [the #9646 convergence report](convergence_report.md) and [its static figure](convergence_constraints_first_lexicographic_v1.png).

Replay provenance was made portable by rewriting worktree-local paths and adding explicit producer-path and exact-copy retention bindings. `path_normalization.json` records the source and final bundle hashes, every rewritten field, and the added binding fields. The preserved source records matched after those path changes and bundle-only annotations were removed; scenario, planner, metric, event, and outcome fields were unchanged. The original provenance JSON files used for this comparison remain in local `output/` and are not part of the durable bundle, so this is a verified transformation of the preserved local artifacts rather than an independently retrievable source package.

## Replays

- Pilot representative `seed_1101/optuna/candidate_0010` was recorded as a task success with 11.118 m minimum human distance and 0 near misses. Its re-run matched the original status, termination, step count, seed, outcome, and selected metrics exactly. Severe-intrusion evidence is absent, so its safety criticality remains unknown. This verifies a persisted successful-case replay, not a discovered failure.
- Historical candidate `issue_1501/failure_0002` was materialized with the existing `write_candidate_inputs` helper from the tracked #1501 archive. At recorded revision `58e516aa4f69ff3098bf518199f483006589758c`, the `goal` planner collided with a pedestrian after 10 steps; the event produced 1 pedestrian collision, 5 near misses, and 1.384 m minimum human distance. The two replay JSONL files differ in timestamps, measured step rate, and wall time. After removing exactly `timestamps.start`, `timestamps.end`, `timing.steps_per_second`, and `wall_time_sec`, their canonical 23,907-byte JSON projections are equal and hash to `5b5047919ac185317eeef78268564a3ed1a9667ddab915ecf1e5d4ddf1aeb1f6`. The prior packet signature `6e34dc5d810fecaea9921934fd6b386e4c21319a3e59d225538fb8b0cb818c00` is retained as an unverified legacy digest. The regenerated scenario’s recorded `scenario_cert.v1` classification is `hard_but_solvable` / `eligible`; its exact historical input binding remains unknown, so it is not admitted to the regression corpus.

## Why the current search missed the known case

The pilot fixed the simulator scenario seed at 123 and clipped `start_x` to a minimum of 2.5 m after its first certification smoke landed in an inflated wall cell. The regenerated #1501 candidate uses scenario seed 320 and `start_x` 2.312 m, so it falls outside the clipped pilot domain. The one-candidate invalid smoke was rejected before planner evaluation and was retained as a separate diagnostic attempt; the corrected one-candidate smoke completed successfully.

The current fixed-seed pilot’s minimum human distance was over 11 m and no near misses appeared. Together with the tied zero scores, the recorded metrics do not show variation that would justify spending a larger budget on the same domain. Severe-intrusion evidence is missing, so this does not establish the absence of critical cases. The #1501 archive used a different objective and varied candidate scenario seeds; those results are context, not a matched comparison.

## Limits and next action

Only the `goal` baseline was evaluated. There was no planner optimization, held-out family, regression-corpus admission, alternate-planner feasibility run, or co-evolution round. Fourteen other archived #1501 failures were not replayed in this work. No candidate is reported as infeasible merely because this planner collided. All 64 exact pilot episode records are retained with producer SHA-256 bindings; their source revision is consistent, but they lack severe-intrusion evidence. The producer `analysis_eligibility.trace_present` field means the episode-record path was present, not that detailed traces were captured. Their `record_simulation_step_trace` and `record_planner_decision_trace` flags are false, and their embedded `scenario_digest` and `map_digest` are null, so these episode summaries are not detailed traces or standalone exact scenario/map bindings. Absolute producer route-override paths remain in the exact source bytes and are not dereferenced by the report. Candidate-specific scenario YAMLs, route overrides, and trajectories are not all retained, so these files do not provide portable replay material for all pilot candidates.

The run command included an execution-context label naming Python 3.12.3. The runner’s machine-generated provenance records Python 3.13.14 and is authoritative; the label was inaccurate and did not control the interpreter.

**Next action:** do not launch #9648. Record NO-GO and close it with this evidence after #9645’s receipt is attached. Keep the regenerated #1501 candidate pending in the counterexample-corpus work until exact historical input binding is established, then define a separate bounded pilot that includes varied environment seeds and does not exclude known admissible cases. Re-review the gate before any larger campaign.

## Machine-readable packet

- `summary.json` — design, budgets, outcomes, and decision.
- `candidate_evaluations.csv` — all 64 pilot candidates with certification, status, metrics, and checksums.
- `best_so_far.csv` — all 16 evaluation points per run.
- `smoke_attempts.csv` — the initial certification rejection and corrected successful smoke, kept outside the 64-candidate budget.
- `run_metadata.json` — exact command, source/config hashes, machine identity, and runtime notes.
- `row_status.json` — fail-closed classification for all 64 candidate executions.
- `replay_validation.json` — exact pilot-success replay and two projection-matched (not byte-identical) collision replays for a candidate regenerated from #1501 archive inputs at recorded revision 58e516aa4f69ff3098bf518199f483006589758c; the comparison projection and digest are reproducible, while historical input binding remains unknown and corpus admission is pending.
- `evidence_synthesis.json` — observed evidence, bounded inference, caveats, and evidence table.
- `source_hashes.sha256` — hashes for the tracked source/config/runtime inputs.
- `convergence_report.json` / `.md` and `convergence_constraints_first_lexicographic_v1.png` — the corrected #9646 schema-v3 report, table, and convergence figure; unknown safety criticality is preserved.
- `pilot_report_manifest_path_map.v1.json` — preserves the original producer manifest-path bindings.
- `pilot_report_artifact_path_map.v1.json` — binds retained comparison, manifests, and all 64 episode records to the producer paths and exact digests.
- `reproduction_inputs/` — derived manifests that rebind only episode-record paths to tracked exact copies, plus the exact comparison-bound manifest map and transformation provenance.
- `source_episode_records/` — all 64 candidate episode-record JSONLs, each matching its producer SHA-256 and byte count; retained to verify per-candidate episode-record summaries and source revision; detailed simulation-step and planner-decision trace capture is disabled for all 64 records.
- `report_provenance.json` — exact schema-v3 report command, input and path-map digests, generator commit, output checksums, and no-rerun declaration.
- `archive/report_schema_v1/` — exact-byte copies and prior provenance for the superseded v1 convergence report.
- `source_manifests/` — all four original 16-evaluation run manifests, retained with the comparison index and per-row checksums.
