# Adversarial replay gallery

Build a small, replay-checked visual bundle from a persisted
`adversarial-search-manifest.v1` file. The gallery reuses the existing failure archive,
canonical benchmark runner, objective registry, and episode replay figures.

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/tools/materialize_adversarial_replay_gallery.py \
  output/<search-run>/manifest.json \
  --out output/adversarial-replay-gallery/<run-name> \
  --top-k 5 --no-video
```

The output directory must be new. Search inputs remain read-only; generated bundles belong under
the ignored `output/` directory. The command writes `gallery_manifest.json`, a compact `README.md`,
and a separate case directory for each selected candidate. Every input candidate stays in the
manifest accounting, including failed evaluations, missing source files, invalid certificates,
duplicates, and candidates below the top-K cutoff.

## Selection and replay checks

The selector requires an analysis-eligible row, a `valid` or `hard_but_solvable` certificate, a
finite objective, one unambiguous source episode, a one-scenario YAML input, matching scenario and
seed identity, candidate parameters matching the generated scenario metadata, and a recomputed
effective-scenario hash matching the search manifest. The source failure attribution must agree
with the canonical episode. Successful
episodes are accounted as `source_episode_not_a_failure` and are not shown as falsification cases.
Selected cases are ranked by objective value, then source candidate index; exact scenario hashes
and existing failure-mechanism clusters reduce duplicate displays.

For each selected row, the tool copies the scenario YAML and its referenced map/route files into
the case bundle, then calls the canonical benchmark runner with the recorded policy and available
search configuration. It records a step trace for visualization. The replay is compared against
the source episode's identity, canonical outcomes, registered objective value, and the configured
absolute tolerance.

`replay_match: match` means those values agree. `verification_status: verified` additionally
requires matching known source and replay revisions. A matching replay at a different revision is
reported as `outcome_reproduced_revision_changed`; missing revision provenance stays explicit.
Mismatches, missing inputs, failed execution, and missing replay records remain visible in the
case manifest.

## Reading the bundle

- `gallery_manifest.json` records source manifest hash/revision, declared method (or `unknown`),
  seed, budget, search-space hash, candidate dispositions, and per-case results.
- `cases/<case-id>/case_manifest.json` records the source links and hashes, source certificate
  classification, copied scenario and file-backed runner configuration, effective runner settings,
  replay comparison, and available rendering outputs.
- `cases/<case-id>/figures/` uses the existing still, filmstrip, and trajectory renderer on the
  canonical replay trace. Each case records video as `rendered`, `unavailable`, `not_attempted`,
  or `disabled`; a request that produces no video file is never reported as successful. The
  current map-backed batch runner does not emit synthetic video for these search scenarios, so
  their case manifests mark requested video as unavailable.

The original source episode JSONL is copied byte-for-byte into each case bundle, so diagnostic
values such as non-finite sentinels are preserved without rewriting source evidence. The source
certificate is carried as source evidence; it is not promoted to a mathematical feasibility proof.
A replay at a changed revision is evidence that the reported outcome was reproduced in that run,
not an exact-source replay. The gallery does not infer a search method from file names, reconstruct
missing models, admit cases into the regression corpus, or establish real-world safety. A run with
zero selected cases is a valid result when no eligible attributed failures were present.
