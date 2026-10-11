# Byte-Exact Baseline Controller Rows

These files contain the first complete written raw episode row per algorithm
present in the captured baseline job 31302 output. Source commit:
`373dbfde4f39667cf9e8732dabe7df5118bdeab1`. All fixture seeds are 1001.

The goal row is the unchanged #10313 reproduction, including its newline:
397583 bytes, SHA-256
`6084b29551a667654d511dad222a96b777910d42e1672aad4617daff707b10ab`.
`manifest.json` records relative source files, byte sizes, identities and exact
digests for all eight rows. The hybrid campaign arm records algorithm
`hybrid_rule_local_planner`; the manifest preserves both identities.

No metadata, optional availability reports, recorded input identities, actions or
trace geometry were scrubbed or regenerated. Tests attach only
`execution_status=written`, as the producer's loader does. Recorded input paths
are inert provenance strings; tests never open them. Fixture replay is audit
implementation evidence, not a new simulation, full-cohort completion, controller
performance claim or independent acceptance of a behaviour receipt.
