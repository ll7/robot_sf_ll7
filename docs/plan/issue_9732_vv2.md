# VV-2 reference-planner oracle plan

## Goal and scope

Implement issue #9732 for the 48-scenario release matrix and seeds 111–140: a blind goal run with no pedestrians, a stationary robot run with the original pedestrian population, and a paired dominance gate for pedestrian-aware planner results. The output is a versioned JSON report and a Markdown summary. Spawn, social-force, and hybrid-planner implementations are out of scope.

## Evidence sources

- Issue #9732 and parent #9730; issue #9729 supplies the stationary-contact interpretation.
- `configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml` pins the matrix, seed set, and horizon.
- `robot_sf/benchmark/map_runner/` owns episode execution and provenance.
- `docs/benchmark_governance.md` defines the evidence and release-claim boundary.

## Steps

1. Register a native `stand_still` reference policy that always commands zero speed and turn.
2. Add a runner config pinning the matrix, seeds, probe declarations, planner arms, and thresholds. Derive the no-pedestrian scenario overlay without changing source maps or spawn code; verify zero instantiated pedestrians.
3. Execute the goal and stationary arms through the map runner and write exact scenario/seed records. Accept optional pedestrian-aware no-pedestrian episode inputs for the dominance comparison; require configured coverage in release-gate mode.
4. Check completeness, outcome and contact semantics, configured thresholds, and paired dominance. Emit JSON and Markdown even when the gate fails.
5. Validate focused behavior and report fixtures, then execute the full matrix on a cluster from a clean exact commit. Preserve raw outputs and receipts durably, and open a PR without merging.

## Decision and stop rules

- A non-probe goal failure, an incomplete or degraded row, a stationary contact rate above its configured bound, or a paired aware-planner regression fails the release gate.
- A contact is an episode with a typed pedestrian contact event; missing typed contact data cannot count as zero.
- The source matrix's intentionally empty `classic_bottleneck_low` rows remain in coverage and are excluded from the conditional stationary-contact denominator by an explicit checked config entry.
- Dominance compares the same scenario and seed under the same pedestrian-free overlay. Missing configured aware rows block the gate; they are never silently skipped.
- Probe exemptions must be explicit in config and remain listed in the report. Failures in probes remain diagnostic findings.
- Route acceptance or local smoke is not benchmark evidence. The cluster result requires complete rows, provenance, and recoverable artifacts before a release claim.

## Validation and recovery

- Run focused tests for stationary commands, zero-population overlay, row completeness, contact accounting, dominance, and gate exit codes.
- Run lint/format and repository PR readiness; audit cluster launcher and exact source before submission.
- Keep the worktree, branch, scheduler identity, run root, and raw outputs on any interruption. Resume only from a recorded matching config and source identity.
