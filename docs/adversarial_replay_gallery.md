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

The output directory must be a new child of this checkout's ignored `output/` directory. The API
resolves paths before checking this boundary, so a symlink that escapes `output/` is rejected too.
Search inputs remain read-only. The command writes `gallery_manifest.json`, a compact `README.md`,
and a separate case directory for each selected candidate. Every input candidate stays in the
manifest accounting, including failed evaluations, missing source files, invalid certificates,
duplicates, and candidates below the top-K cutoff.

## Lightweight CLI smoke

When a search pilot produces no failure rows, use the one-candidate historical compatibility
fixture at `tests/fixtures/adversarial_replay_gallery/issue_1501_compat/manifest.json` to exercise
the real runner and renderer. It is adapted from the tracked #1501 `failure_0002` case and the
#9645 replay; it is not a persisted search run or a new discovery. Run it with a fresh output path:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/tools/materialize_adversarial_replay_gallery.py \
  tests/fixtures/adversarial_replay_gallery/issue_1501_compat/manifest.json \
  --out output/adversarial-replay-gallery/<smoke-name> --top-k 1
```

The smoke should select and replay one candidate. Compare the recorded outcome and objective, then
inspect `gallery_manifest.json`, the case manifest, and `figures/`. The fixture's dynamic task
feasibility is unknown. A current-code replay at a different source revision is an outcome
reproduction only; video may be recorded as unavailable when the canonical runner emits no video.
The output stays in ignored `output/`; retain only a compact checksummed receipt when a durable
handoff is needed.

## Selection and replay checks

The selector requires an analysis-eligible row, a `valid` or `hard_but_solvable` certificate, a
finite objective, one unambiguous source episode, a one-scenario YAML input, matching scenario and
seed identity, candidate parameters matching the generated scenario metadata, and a recomputed
effective-scenario hash matching the search manifest. The source failure attribution must agree
with the canonical episode. Source availability must explicitly report `available`, native
readiness, and native execution mode in both the attribution and eligibility receipts; the source
episode must also report successful algorithm metadata with no fallback/degraded runtime marker.
Missing, fallback, degraded, or inconsistent source availability stays in candidate accounting and
cannot be selected as a critical discovery. Successful episodes are accounted as
`source_episode_not_a_failure` and are not shown as falsification cases.
The fallback scan checks algorithm metadata but ignores unsupported statuses in the paired-metric
and simulation-step-trace diagnostic products. Those statuses do not by themselves mean planner
execution fell back; explicit fallback or degraded markers remain disqualifying.
Selected cases are ranked by objective value, then source candidate index; exact scenario hashes
and existing failure-mechanism clusters reduce duplicate displays.

For each selected row, the tool copies the scenario YAML and its referenced map/route files into
the case bundle, then calls the canonical benchmark runner with the recorded policy and available
search configuration. For `map_id` scenarios, selection snapshots the resolved map and registry
bytes, materializes both, and points the runner at a derived registry that resolves the same ID to
the bundled map. The source binding also checks the map path recorded in the source episode and the
tracked registry/map bytes. If the source episode does not attest the registry digest, or the
registry came from an external override, the binding stays `unknown` and cannot become `verified`.
Scenarios that rely on the implicit default map pool without an explicit `map_file` or resolved
`map_id` snapshot are also `unknown`; the gallery does not infer their map identity from the
planner's outcome. It records a step trace for visualization. The replay is compared against the
source episode's identity, canonical outcomes, registered objective value, and the configured
absolute tolerance.

`replay_match: match` means identity, exact categorical outcome/failure attribution, and objective
projection agree and the canonical runner reports an available, successful replay. Matching
objective projections cannot hide a different termination reason, event flag, or primary failure.
Fallback, skipped, failed, missing, or inconsistent runner availability keeps the case `unavailable`,
even when the other comparisons agree; those diagnostics and the runner summary remain in the case
manifest.

`verification_status: verified` additionally requires a known source revision from the episode
record or, if absent there, from the manifest. If both provide a revision, they must agree. The
replay revision and clean gallery code checkout must match that exact source revision; source
map/registry/config files must match tracked files at that revision; the source episode must attest
the map registry digest used by a `map_id` input; and source/replay planner configuration hashes
must match. A matching replay without complete input binding remains
`outcome_reproduced_source_inputs_unbound`; a dirty or different checkout has its own explicit
status. A matching replay at another revision is reported as
`outcome_reproduced_revision_changed`; missing revision provenance stays explicit. Input mismatches,
failed execution, and missing replay records remain visible in the case manifest. `replay_match` is
therefore an outcome comparison; only `verification_status: verified` asserts exact-source replay
verification under these recorded checks.

## Reading the bundle

- `gallery_manifest.json` records source manifest hash/revision, declared method (or `unknown`),
  seed, budget, search-space hash, candidate dispositions, and per-case results.
- `cases/<case-id>/case_manifest.json` records the source links and hashes, source certificate
  classification, copied scenario and file-backed runner configuration, effective runner settings,
  replay comparison and input-binding checks, and available rendering outputs.
- `cases/<case-id>/figures/` uses the existing still, filmstrip, and trajectory renderer on the
  canonical replay trace and passes a materialized map only after the same image reader used by the
  renderer can decode it. The case manifest records `map_context: overlay_rendered` only after that
  preflight and render succeed; otherwise it records the source map path/digest and an exact
  unavailable reason (for example, the existing image reader cannot decode an SVG). A map path
  being present no longer implies that an overlay was drawn.
  Pedestrian trajectory lines require stable actor IDs. Unknown identities are rendered as isolated
  positions. When an actor ID reappears with a displacement above the documented 12 m/s
  visualization continuity threshold, the track is split into segments and the split count is
  recorded. This is a rendering safeguard, not a feasibility or physics verdict.
  Each case records video as `rendered`, `unavailable`, `not_attempted`, or `disabled`; a request
  that produces no video file is never reported as successful. The current map-backed batch runner
  does not emit synthetic video for these search scenarios, so their case manifests mark requested
  video as unavailable.

The original source episode JSONL is copied byte-for-byte into each case bundle, so diagnostic
values such as non-finite sentinels are preserved without rewriting source evidence. The source
certificate is carried as source evidence; it is not promoted to a mathematical feasibility proof.
A replay at a changed revision is evidence that the reported outcome was reproduced in that run,
not an exact-source replay. The gallery does not infer a search method from file names, reconstruct
missing models, admit cases into the regression corpus, or establish real-world safety. A run with
zero selected cases is a valid result when no eligible attributed failures were present.
