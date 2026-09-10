# Issue #8571: Default-disabled bounded falsification slice

The checked-in issue #8570 packet defines the first station-platform falsification design. This
follow-up composes that packet with the existing adversarial search-harness and materialization
owners to produce a deterministic preflight ledger. It does not authorize or perform native
compute.

Run the preflight from a linked worktree with:

```bash
uv run python scripts/adversarial/run_issue_8571_bounded_falsification_slice.py \
  --output output/adversarial/issue_8571/preflight.json
```

The default exit code means only that the diagnostic ledger was generated. Use
`--fail-on-blocked` when a caller needs a blocked gate to be a nonzero admission
result; it returns status `2` for the current packet and still does not authorize
compute.

The `--packet` argument remains supported for the committed canonical path. Preflight rejects
other paths and verifies both the committed packet file digest and the packet's trusted
`self_digest` before loading it.

The report source-hashes the packet inputs, checks the typed bounds and pedestrian binding,
proves zero-overlay source equivalence, and prepares 64-candidate random and Halton control arms
for each declared search seed. Candidate and pre-simulation rejection records retain immutable
manifest, overlay, and seed identities. The declared CMA-ES arm is recorded but not instantiated.
Generated route payloads remain explicit `proposal_only` overlays: they carry a payload digest,
but no `route_overrides_file` is claimed until the existing bundle writer materializes an artifact.
The canonical scenario runtime builder rejects an inline proposal instead of silently ignoring it.

In the current packet, every overlay-ready row is `blocked` because `pedestrian_delay_s` is not
runtime-effective in template mode and native/replay producers are unavailable. Rows are never
converted to `result` or `null`; native outcome and replay digests remain absent until the packet
gate, same-fixture replay/no-op fidelity, and durable artifact custody are independently cleared.
The output is diagnostic preparation, not benchmark, safety, robustness, or scientific evidence.
