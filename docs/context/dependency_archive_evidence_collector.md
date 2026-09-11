# Dependency Archive Evidence Collector

The standalone `scripts/tools/collect_dependency_archive_evidence.py` command
consumes an exact `robot_sf.p04.p05_batch.v1` manifest and records target
selection, PyPI release metadata, archive size/SHA-256 checks, and bounded
metadata/license/notice-member observations. It is a supporting evidence
collector, not a replacement for the strict inventory or rights policy.

```bash
python scripts/tools/collect_dependency_archive_evidence.py \
  --task-id <task-id> \
  --output output/validation/dependency-archive-evidence \
  --batch-manifest <manifest.json> \
  --expected-batch-sha256 <sha256> \
  --expected-owner-issue <issue> \
  --expected-audit-source-sha <sha1> \
  --expected-candidate-commit-sha <sha1> \
  --expected-candidate-tree-sha <sha1>
```

Use `--offline` for a cache-only replay. Missing, malformed, ambiguous, or
resource-limited rows remain `partial_or_unavailable`; a diagnostic run may
still exit `0` while retaining those unresolved rows. Invalid input or output
failures exit non-zero. Registry archive URLs must remain HTTPS on the public
PyPI file hosts, and every redirect hop is rejected if it leaves those hosts.
The input manifest is capped at 16 MiB and 50,000 members; the durable registry
observation keeps only identity, license-descriptor, project-link, and
release-file routing fields from the bounded response. Target wheel selection
matches Python, operating-system, and architecture tags; source distributions
(including ZIP sdists) are inspected separately, and registry matches require
the corresponding PyPI `packagetype`.

The durable `dependency_evidence_ledger.json` uses output-relative cache paths
and is reproducible for identical manifest, cache, and registry observations.
`private_cache_summary.json` classifies the successfully used archive bytes as
ignored local cache and does not recursively hash unrelated stale cache files;
`collector-run.json` is volatile run metadata and its timestamp is not evidence.
Neither the collector nor this note treats package metadata, archive text, or
manifest routing as permission, redistribution, release, benchmark, or
scientific evidence.
