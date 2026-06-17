# Issue 1470 Oracle Imitation Trace Closeout

This bundle promotes the retrieved Slurm evidence for job `12911`, submitted from
`robot_sf_ll7@dcb14927f08277d123d9666ad91b8d6abbc4fe9d`, into a small tracked
repository artifact.

## Result

- Slurm job: `12911`
- Private ops ledger label: `issue1470-oracle-imitation`
- Candidate: `hybrid_rule_v3_static_margin0_waypoint2`
- Split/profile: `train` / `testing`
- Episodes: `6`
- Success rate: `0.6666666666666666`
- Collision rate: `0.16666666666666666`
- Near-miss rate: `0.16666666666666666`
- Termination reasons: `success=4`, `collision=1`, `max_steps=1`
- Scenario families: `classic=4`, `francis2023=1`, `nominal=1`

## Claim Boundary

This is durable trace-collection evidence for #1470/#2441. It proves the selected
launch-packet source candidate ran on the selected train split and preserves the
small JSONL episode evidence for downstream imitation dataset materialization.

It is not final imitation training evidence, not a final NPZ dataset promotion, and
not benchmark-quality planner promotion evidence.

## Files

- `summary.json`: compact closeout summary and claim boundary.
- `oracle_imitation/issue1470_train_candidate_traces/oracle_candidate_trace_manifest.json`:
  source manifest emitted by the trace collection job.
- `oracle_imitation/issue1470_train_candidate_traces/train__hybrid_rule_v3_static_margin0_waypoint2__combined.jsonl`:
  six collected episode records.
- `source_slurm_checksum_manifest.sha256`: checksum manifest from the retrieved Slurm output.
- `SHA256SUMS`: checksums for this tracked bundle.

## Validation

```bash
python3 -m json.tool docs/context/evidence/issue_1470_oracle_imitation_traces_12911_2026-06-17/summary.json >/dev/null
python3 -m json.tool docs/context/evidence/issue_1470_oracle_imitation_traces_12911_2026-06-17/oracle_imitation/issue1470_train_candidate_traces/oracle_candidate_trace_manifest.json >/dev/null
(cd docs/context/evidence/issue_1470_oracle_imitation_traces_12911_2026-06-17 && sha256sum -c SHA256SUMS)
```
