---
release_tag: "v0.0.0"
release_commit: "0000000000000000000000000000000000000000"
artifact_bundle:
  archive_path: "output/release_bundle.tar.gz"
  doi: "10.5281/zenodo.00000"
  checksum_manifest_path: "output/release_bundle/checksums.sha256"
scenario_matrix:
  path: "configs/scenarios/classic_interactions_francis2023.yaml"
  hash: "000000000000"
seed_policy:
  mode: "fixed"
  seed_set: "default"
  seeds: []
reproduction:
  environment_setup_command: "uv sync --all-extras"
  functional_smoke_command: "python scripts/validation/run_release_functional_badge_smoke.py --bundle-path output/release_bundle"
  headline_reproduction_command: ""
  tolerances: {}
known_nondeterminism: []
expected_unavailable_degraded_rows: []
claimed_badge_level: "available" # none | available | functional | reproduced
justification: "Artifact bundle has been prepared and checksummed."
---

# Release Reproducibility Checklist

Fill in the metadata in the front matter block above for each release.

## Checklist Verification Guide

Use this checklist to self-assess the claimed reproducibility badge level:

### For `available` badge
- [ ] Release tag is created on GitHub.
- [ ] Code commit matches `release_commit`.
- [ ] Release bundle is archived (`.tar.gz`) and uploaded as a release asset or registered with a DOI.
- [ ] Checksums file `checksums.sha256` lists SHA-256 for all exported payload files.

### For `functional` badge
- [ ] All `available` checklist items are checked.
- [ ] Setup/install command runs successfully in a clean virtualenv.
- [ ] Smoke test command runs green using only files within the artifact bundle.

### For `reproduced` badge
- [ ] All `functional` checklist items are checked.
- [ ] Full campaign command executes and regenerates the headline metric tables/plots.
- [ ] Metrics match the published tables within the stated tolerances.
