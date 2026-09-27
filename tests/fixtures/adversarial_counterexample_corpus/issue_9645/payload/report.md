# Issue #9645 bounded falsification pilot

**Evidence boundary:** diagnostic local nominal execution over generated stress scenarios. This is not paper-facing benchmark evidence, a planner ranking, or real-world safety evidence. A finite search with no new counterexample does not establish that none exists.

## Decision

**NO-GO for scaling #9648 under this exact search-space and fixed-environment design.** All 64 planned evaluations completed with recorded task success, zero collisions, timeouts, route failures, and near misses; minimum clearance was at least 9.718 m, and all four best-so-far curves stayed at 0.0. Severe-intrusion evidence is absent for every candidate, so safety criticality is unknown for all 64. These observations do not justify scaling this unchanged design; they do not establish that the cases were non-critical.

This NO-GO is specific to the current fixed-seed domain. It does not say the falsification system cannot find counterexamples: the tracked #1501 archive already contains 15 collision failures under a different search contract. One archived goal-planner case was regenerated from its tracked parameters and replayed twice at recorded source revision `58e516aa4f69ff3098bf518199f483006589758c`; both replays produced the same pedestrian collision. That case has a static `hard_but_solvable` certificate, while dynamic task feasibility remains unknown because no successful reference-planner execution was tested. No replay at the current PR head is claimed.

## Pilot results

| Sampler | Sampler seed | Evaluations | Valid / invalid / failed | Best score | Collisions | Near misses | Minimum clearance range (m) | Safety criticality | Episode timestamp span (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| optuna | 1101 | 16 | 16 / 0 / 0 | 0.0 | 0 | 0 | 9.718–11.169 | unknown (16/16) | 6.576 (partial) |
| random | 1101 | 16 | 16 / 0 / 0 | 0.0 | 0 | 0 | 9.809–10.746 | unknown (16/16) | 12.091 (partial) |
| optuna | 2202 | 16 | 16 / 0 / 0 | 0.0 | 0 | 0 | 9.831–11.052 | unknown (16/16) | 6.586 (partial) |
| random | 2202 | 16 | 16 / 0 / 0 | 0.0 | 0 | 0 | 9.814–11.048 | unknown (16/16) | 6.569 (partial) |

The overall command took 39.10 seconds wall time. Per-run episode spans are partial intervals from the first episode start to the last episode end, excluding search setup and certification overhead; the run manifests do not record exact per-run driver durations. There were 64 unique candidate specs and 64 unique effective-scenario hashes.


## Convergence summary

The corrected #9646 schema-v3 report was generated from the same comparison index and four digest-pinned archived search manifests in a clean evidence root; report generation did not run a simulator. Each run records 16/16 native, available, scored evaluations with no invalid, failed, scoreless, missing, or duplicate rows. The report classifies every candidate's criticality and collision/severe-intrusion tier as unknown because severe-intrusion evidence is absent. The 64 raw episode records are not in the tracked packet, so byte-verified trace eligibility is 0/64 and the report excludes both seed pairs from eligible paired summaries. The stored score rows remain all 0.0, but the v1 scorer can produce its negative composite when one component is absent; those historical scores do not establish a negative safety tier. No inferential test was performed. The run-level experiment commit `58e516aa4f69ff3098bf518199f483006589758c` is recorded in `run_metadata.json`; the report's trace-derived source-revision status is unknown without the raw episode records. Search-level duration is absent from the run manifests; the separate 39.10-second shell wall time is recorded in `run_metadata.json`.

See [the #9646 convergence report](convergence_report.md) and [its static figure](convergence_constraints_first_lexicographic_v1.png).

Replay records were made portable by normalizing only worktree-local route, schema, and command paths. The original file hashes, normalized hashes, and rewritten fields are recorded in `path_normalization.json`; all non-path record fields were compared with the captured source outputs.

## Replays

- Pilot representative `seed_1101/optuna/candidate_0010` was recorded as a task success with 11.118 m minimum human distance and 0 near misses. Its re-run matched the original status, termination, step count, seed, outcome, and selected metrics exactly. Severe-intrusion evidence is absent, so its safety criticality remains unknown. This verifies a persisted successful-case replay, not a discovered failure.
- Historical `issue_1501/failure_0002` was materialized with the existing `write_candidate_inputs` helper from the tracked #1501 archive. At recorded revision `58e516aa4f69ff3098bf518199f483006589758c`, the `goal` planner collided with a pedestrian after 10 steps; the event produced 1 pedestrian collision, 5 near misses, and 1.384 m minimum human distance. Two replays at that revision had identical comparison signatures. The recorded `scenario_cert.v1` classification is `hard_but_solvable` / `eligible`.

## Why the current search missed the known case

The pilot fixed the simulator scenario seed at 123 and clipped `start_x` to a minimum of 2.5 m after its first certification smoke landed in an inflated wall cell. The replayed historical case uses scenario seed 320 and start_x 2.312 m, so it falls outside the clipped pilot domain. The one-candidate invalid smoke was rejected before planner evaluation and was retained as a separate diagnostic attempt; the corrected one-candidate smoke completed successfully.

The current fixed-seed pilot’s minimum human distance was over 11 m and no near misses appeared. Together with the tied zero scores, the recorded metrics do not show variation that would justify spending a larger budget on the same domain. Severe-intrusion evidence is missing, so this does not establish the absence of critical cases. The #1501 archive used a different objective and varied candidate scenario seeds; those results are context, not a matched comparison.

## Limits and next action

Only the `goal` baseline was evaluated. There was no planner optimization, held-out family, regression-corpus admission, alternate-planner feasibility run, or co-evolution round. Fourteen other archived #1501 failures were not replayed in this work. No candidate is reported as infeasible merely because this planner collided. The 64 raw pilot episode records remain outside the tracked packet; their absence limits trace review and byte-verified comparison eligibility.

The run command included an execution-context label naming Python 3.12.3. The runner’s machine-generated provenance records Python 3.13.14 and is authoritative; the label was inaccurate and did not control the interpreter.

**Next action:** do not launch #9648. Record NO-GO and close it with this evidence after #9645’s receipt is attached. Add the replay-verified historical case to the counterexample-corpus work, then define a separate bounded pilot that includes varied environment seeds and does not exclude known admissible cases. Re-review the gate before any larger campaign.

## Machine-readable packet

- `summary.json` — design, budgets, outcomes, and decision.
- `candidate_evaluations.csv` — all 64 pilot candidates with certification, status, metrics, and checksums.
- `best_so_far.csv` — all 16 evaluation points per run.
- `smoke_attempts.csv` — the initial certification rejection and corrected successful smoke, kept outside the 64-candidate budget.
- `run_metadata.json` — exact command, source/config hashes, machine identity, and runtime notes.
- `row_status.json` — fail-closed classification for all 64 candidate executions.
- `replay_validation.json` — exact pilot-success replay and two identical historical collision replays at the recorded source revision 58e516aa4f69ff3098bf518199f483006589758c.
- `evidence_synthesis.json` — observed evidence, bounded inference, caveats, and evidence table.
- `source_hashes.sha256` — hashes for the tracked source/config/runtime inputs.
- `convergence_report.json` / `.md` and `convergence_constraints_first_lexicographic_v1.png` — the corrected #9646 schema-v3 report, table, and convergence figure; unknown safety criticality is preserved.
- `pilot_report_manifest_path_map.v1.json` — binds the comparison's four original manifest paths to checksum-pinned archived manifests.
- `report_provenance.json` — exact schema-v3 report command, input and path-map digests, generator commit, output checksums, and no-rerun declaration.
- `archive/report_schema_v1/` — exact-byte copies and prior provenance for the superseded v1 convergence report.
- `source_manifests/` — all four original 16-evaluation run manifests, retained with the comparison index and per-row checksums.
