# Issue #1318 TEB Corridor-Deadlock Evidence

Date: 2026-05-20

This bundle preserves a compact summary of the local #1318 corridor-deadlock benchmark slice.
Raw JSONL episode records remain in ignored `output/benchmarks/` because the run is reproducible
from the tracked scenario slice, commands, and commit.

Tracked evidence:

- `summary.json`: per-algorithm/per-scenario aggregate outcomes for TEB, ORCA, and the hybrid-rule
  incumbent on `configs/scenarios/sets/issue_1318_teb_corridor_deadlock_slice.yaml`.

Raw local outputs used to produce the summary:

- `output/benchmarks/issue_1318_teb_corridor_deadlock_teb.jsonl`
- `output/benchmarks/issue_1318_teb_corridor_deadlock_orca.jsonl`
- `output/benchmarks/issue_1318_teb_corridor_deadlock_hybrid.jsonl`
