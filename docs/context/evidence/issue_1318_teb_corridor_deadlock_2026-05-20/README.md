# Issue #1318 TEB Corridor-Deadlock Evidence

Date: 2026-05-20

This bundle preserves a compact summary of the local Issue #1318 corridor-deadlock benchmark slice.
Raw JSONL episode records remain in ignored local benchmark output because the run is reproducible
from the tracked scenario slice, commands in the context note, and commit.

Tracked evidence:

- `summary.json`: per-algorithm/per-scenario aggregate outcomes for TEB, ORCA, and the hybrid-rule
  incumbent on `configs/scenarios/sets/issue_1318_teb_corridor_deadlock_slice.yaml`.

Raw local outputs used to produce the summary were disposable ignored benchmark JSONL files for the
TEB, ORCA, and hybrid-rule runs. They are intentionally not referenced as durable dependencies.
