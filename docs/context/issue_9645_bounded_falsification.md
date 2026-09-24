# Bounded falsification pilot

The pilot comparison is persisted in
`docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/pilot_comparison.json`.
Its report can be regenerated from that JSON without running a search, replay, or simulator:

```bash
uv run python scripts/tools/compare_adversarial_samplers.py \
  --repo-root . \
  --render-existing-json path/to/pilot_comparison.json \
  --render-execution-mode empirical \
  --out-md path/to/pilot_comparison.md \
  --render-provenance-json path/to/pilot_comparison_render_provenance.json
```

The execution mode is an explicit caller declaration. The renderer validates the stored comparison,
preserves its recorded candidate outcomes, and writes the output and renderer provenance; it does
not infer empirical execution from the input or repeat search and simulation. The provenance records
the source and rendered report digests, renderer revision, and `search_or_simulation_rerun: false`.

The experiment's recorded source commit, `58e516aa4f69ff3098bf518199f483006589758c`, is local-only
and cannot currently be fetched from the remote. Its exact experiment-time revision can be
reconstructed from base `5cccee50be333adceee4c978b54bf63d32454cc9` and the tracked
[`source_revision_58e.patch`](evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_revision_58e.patch);
the patch digest, reconstruction commands, and verified file list are recorded in
[`source_revision_mapping.json`](evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_revision_mapping.json).
All 11 declared source-file hashes match both the reconstructed revision and public commit
`0ed1f06b5a24892e7409e653579ae35733b5d79d`. This establishes identity for those 11 files only,
not full-tree equivalence. The source patch was applied to the declared base and all 11 checksums
verified successfully.

For the pilot, the bounded Random/TPE comparison produced no critical candidate in four runs at 16
evaluations each. That is a finite-budget null result and a **NO-GO** for the scaled campaign in
issue #9648; it does not establish that the scenario space has no counterexample.
