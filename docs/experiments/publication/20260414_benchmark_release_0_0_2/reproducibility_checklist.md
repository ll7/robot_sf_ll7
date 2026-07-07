---
claimed_badge_level: functional
artifact_bundle:
  archive_path: https://github.com/ll7/robot_sf_ll7/releases/download/0.0.2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz
  checksum_manifest_path: docs/experiments/publication/20260414_benchmark_release_0_0_2/release_metadata.json
  doi: 10.5281/zenodo.19563812
reproduction:
  functional_smoke_command: uv run pytest tests/benchmark/test_paper_results_handoff.py::test_canonical_handoff_matches_durable_release_campaign_table
---

# Reproducibility Checklist for Benchmark Release 0.0.2

This checklist documents the reproducibility status of the `0.0.2` release bundle, verifying that it is functional and meets the badging criteria.

## Rubric Self-Assessment

- [x] **available**: The release bundle is published, hash-pinned, and has a durable DOI.
- [x] **functional**: The bundle is self-sufficient. Headline results can be re-derived.
