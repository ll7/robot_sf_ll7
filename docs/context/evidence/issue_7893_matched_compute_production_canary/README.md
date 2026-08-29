# Matched-Compute Production Canary Receipt — issue #7893

**Status:** blocked before arm execution — the reactive production driver is absent, so the canary
does not start either arm or leave a partial comparison artifact. Controller-snapshot provenance is
explicitly not fabricated.
**Issue:** [#7893](https://github.com/ll7/robot_sf_ll7/issues/7893) (parent #4360; frozen packet
#6921).
**Entry point:** `scripts/validation/run_matched_compute_production_canary.py`.

## Execution

- Repository commit: `4b57a26525c74550624c3523d5420c93b29aae78`
- Packet: `configs/adversarial/issue_6921_matched_compute_packet.yaml`
- Packet digest (SHA-256): `4527f00dc700409bf9d9f9c0c2a1d0958903693368afd789793415726057b2f3`
- Frozen input digest lock: `input_digests.json` records the packet and every recursively
  referenced YAML configuration digest. The runner derives the reference closure from the packet,
  rejects unsafe or ambiguous paths, and verifies every byte before either arm can start.
- Positive receipts also emit one `matched_compute_trace.v1` runtime trace per arm. The runner
  reconciles each trace's accepted/rejected/invalid partition against the candidate records,
  binds candidate identity fields to the frozen packet arm, and requires each trace budget to
  equal the packet's 90-candidate per-episode budget. `--check` also requires non-empty
  frozen-input digests plus a source commit present in the current checkout and reachable from
  its `HEAD`.
- Command: `python scripts/validation/run_matched_compute_production_canary.py --packet
  configs/adversarial/issue_6921_matched_compute_packet.yaml --output-dir
  output/matched_compute_canary`

## Results

| Arm | Candidates | accepted | invalid | failed | unavailable | Observed steps |
| --- | --- | --- | --- | --- | --- | --- |
| open_loop | 0 | 0 | 0 | 0 | 0 | 0 |
| reactive | 0 | 0 | 0 | 0 | 0 | 0 |

- Pre-execution gate: the frozen `FiniteGridSearchPolicy` + `BoundedResidualAdversary` classes
  resolve, but no native environment-episode driver exposes per-residual candidate evaluations
  and observed simulator-step provenance. This is the exact missing hook required by #7893.
- Because the gate fails before arm execution, both arms have zero candidate records and zero
  observed simulator steps. The open-loop runner is not invoked, so this receipt cannot be read as
  a partial arm comparison.
- Controller-snapshot provenance (`ReactiveRuntimeSnapshot` / one-step probe) remains explicitly
  excluded from production evidence.

## Evidence boundary

- `evidence_status: blocked` — no arm is `production_observed` and no arm began execution.
- No planner ranking, stress-strength, matched-objective-equivalence, safety, realism, benchmark,
  release, dissertation, or publication claim is made.
- Raw per-candidate output lives under `output/matched_compute_canary/` (ignored, worktree-local).
- `--check` mode validates a receipt deterministically (packet-bound budget and runtime-trace
  reconciliation, frozen arm identity fields, duplicate and cross-arm identity checks,
  packet/config digest and reachable source-commit identity, and no fallback/unavailable in
  production-observed arms).

## Next smallest step

A successor slice must add a real environment-episode driving loop for the reactive arm (one
frozen episode through the Robot SF simulator with the packet-bound residual adversary config,
observing each macro-action boundary and residual-candidate evaluation from actual simulator
transitions). It must also provide the candidate-level and episode-level provenance required to
admit either arm as `production_observed`.
