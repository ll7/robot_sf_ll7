# Issue #3207 Simulator-Dependence Validity-Boundary Packet

This packet checks whether existing #3207 fidelity-sensitivity evidence is ready to support a
simulator-dependence validity-boundary claim. It does not run a simulator study or promote any
benchmark, simulator-realism, sim-to-real, paper-facing, or dissertation claim.

## Source Input

- Source summary:
  `docs/context/evidence/issue_3207_fidelity_sensitivity_actual_slice_2026-06-23/summary.json`
- Checker:
  `scripts/validation/check_simulator_dependence_validity_boundary.py`
- Packet:
  `decision_packet.json`

## Decision

- `decision`: `no_claim`
- `evidence_status`: `not_benchmark_evidence`
- `claim_ready`: `false`
- `boundary_violations`: `[]`

Current actual slice is not a full fixed-scope study, and its rank evidence remains
non-identifiable because the primary metric had zero variance. The source summary now carries
explicit no-claim phrases, but downstream reports still must treat this packet as no-claim evidence
because the study scope and rank-identifiability gates do not pass.

## Reproduction

```bash
uv run python scripts/validation/check_simulator_dependence_validity_boundary.py \
  --expected-axis integration_timestep__dt_0_05 \
  --expected-axis social_force_speed_archetypes__mixed_balanced \
  --expected-axis observation_noise__pose_heading_low \
  --expected-axis clearance_radius__radius_0_30 \
  --out docs/context/evidence/issue_3207_simulator_dependence_validity_boundary_packet_2026-06-29/decision_packet.json
```
