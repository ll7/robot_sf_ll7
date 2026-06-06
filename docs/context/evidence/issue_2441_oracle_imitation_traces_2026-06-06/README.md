# Issue #2441 Oracle-Imitation Trace Collection

Date: 2026-06-06

This bundle preserves compact evidence for the Issue #2441 follow-up to parent Issue #1470. It
records two completed SLURM trace-collection jobs:

| Job | Partition | Split | State | Rows | Success | Collision | Near Miss |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `12762` | `a30` | `train` | `COMPLETED` | 6 | 0.6667 | 0.1667 | 0.1667 |
| `12763` | `l40s` | `validation` | `COMPLETED` | 3 | 0.3333 | 0.0000 | 0.0000 |

The source candidate was `hybrid_rule_v3_static_margin0_waypoint2`, collected through
`configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml`,
`scripts/dev/sbatch_oracle_imitation_traces_issue1470.sh`, and
`SLURM/Auxme/issue_1470_oracle_imitation_traces.sl` at commit
`a9679e1a37495b25c1786917fcf2fa7e749d1475`.

## Interpretation

The jobs completed and produced the expected local manifests and combined JSONL trace rows. This is
useful trace-collection and split-routing evidence, but not final imitation-dataset evidence. The
classic slices remain weak:

- train classic: 4 episodes, success `0.5`, collision `0.25`, near miss `0.25`;
- validation classic: 2 episodes, success `0.0`, collision `0.0`, near miss `0.0`;
- validation failures include `bottleneck_yield_failure` and `timeout_low_progress`.

The source candidate can produce trace rows, but the observed classic-slice failures mean these
results should be treated as diagnostic dataset-prep evidence, not proof of a high-quality oracle.

## Split Check

The manifests preserve the launch-packet split contract:

- train seeds: `201, 202, 203, 204, 205, 206`;
- validation seeds: `101, 102, 103`;
- evaluation seeds: `111, 112, 113`.

Only train and validation were collected in this queue-fill pass. Evaluation rows remain
uncollected, so no evaluation examples leaked into the tracked train/validation evidence.

## Artifact Decision

Tracked here:

- `summary.json`: compact synthesis, split check, metrics, and validation commands.
- `finalization_12762.json` / `finalization_12762.md`: required-artifact checksums for the train
  job.
- `finalization_12763.json` / `finalization_12763.md`: required-artifact checksums for the
  validation job.
- `SHA256SUMS`: checksums for this tracked evidence bundle.

Not tracked:

- raw combined trace JSONL files under `output/slurm/...`;
- raw SLURM stdout/stderr logs;
- temporary `/tmp/luttkule/issue1470-*` job work directories.

The raw JSONL traces are still local-only. Downstream imitation training must not treat them as
durable inputs until the manifests and JSONL are uploaded to a durable artifact store or otherwise
given concrete retrieval pointers.

## Validation

```bash
sacct -j 12762,12763 --format=JobID,JobName%24,State,ExitCode,Partition,Elapsed,Start,End -P
jq empty docs/context/evidence/issue_2441_oracle_imitation_traces_2026-06-06/summary.json
sha256sum -c docs/context/evidence/issue_2441_oracle_imitation_traces_2026-06-06/SHA256SUMS
```

Result classification: `completed_pending_artifact_promotion`.
