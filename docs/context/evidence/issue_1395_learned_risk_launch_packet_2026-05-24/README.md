# Issue #1395 Learned Risk Launch Packet Evidence

This directory contains small tracked fixtures for the learned-risk-model launch-packet validator.
They are not training traces, benchmark outputs, or promotion evidence.

Files:

- `trace_contract_fixture.jsonl`: two-row JSONL fixture proving the required trace fields and
  labels can be validated before SLURM training.
- `baseline_summary_stub.json`: compact baseline-freeze record for
  `hybrid_rule_v3_static_margin0_waypoint2`.

Checksums:

```text
178b90c90f6a089bd9fd1d2d7bacc6289c65f8ce3cbde8383f39503a88dff168  trace_contract_fixture.jsonl
1b8598dced652de2983d8919e29e22e9b48905049e30ee5340d381e8e8dc89f2  baseline_summary_stub.json
```
