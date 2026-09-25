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

For a compact multi-run **#9645 evidence packet** whose candidate-specific scenario and episode
files are absent, pass its `payload/` directory to the same command. Directory mode reconciles the
source search manifests against `candidate_evaluations.csv`, `row_status.json`, `summary.json`, and
`convergence_report.json`. The payload must sit beside `evidence_bundle_manifest.json` and
`checksums.sha256`; the command validates their file lists, sizes, and digests before reconciling
rows. It requires `run_metadata.manifest_files` and each summary run's path and digest to name
exactly the loaded source-manifest set. When a manifest inventory includes `artifact_path`, that
locator must resolve inside the packet to the corresponding source-manifest bytes. Candidate-level
errors must agree with the source manifest;
failed or unresolved evaluations stay visible and cannot establish successful execution or a
zero-critical result. Certification counts require a passed certificate with an admissible
classification, and collision counts must agree with the collision-event flag. It binds planned
budgets to per-run manifests, summaries, and convergence counts, including failed, invalid, missing,
and duplicate evaluations. It reports a zero-critical result only when the
declared budget is complete, no candidate is scoreless, every candidate has known noncritical
criticality, and sampler aggregates agree. Unknown criticality, scoreless candidates, and missing
budget slots stay visible and are not counted as zero. A packet containing a critical candidate
requires its original search manifest for normal selection, materialization, and replay. Each packet
row keeps execution outcome, scenario eligibility, case criticality, and replay-input availability as
separate statuses. Missing raw inputs remain missing and do not mean the scenario is infeasible. The
input status covers the candidate's scenario YAML and episode record; it does not certify that all
maps or runner configuration needed for replay are present.

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/tools/materialize_adversarial_replay_gallery.py \
  docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload \
  --out output/adversarial-replay-gallery/issue-9645-accounting
```

This packet result is bounded to its recorded search budget. A zero-critical result does not mean
that no counterexample exists, and historical compatibility replays such as #1501 remain separate
from candidates counted in the packet.

The output directory must be a new child of this checkout's ignored `output/` directory. The API
resolves paths before checking this boundary, so a symlink that escapes `output/` is rejected too.
Search inputs remain read-only. The command writes `gallery_manifest.json`, a compact `README.md`,
and a separate case directory for each selected candidate. Every input candidate stays in the
manifest accounting, including failed evaluations, missing source files, invalid certificates,
duplicates, and candidates below the top-K cutoff.

For a direct complete search run, `summary.num_candidates`, the candidate-list length, and
`config.budget` must agree; a missing summary or truncated inventory is rejected before replay. A
historical compatibility input may omit the producer summary only when it explicitly says it is not
a persisted search manifest. Its gallery marks the inventory incomplete and limits top-K to the
candidate rows supplied; it cannot represent the original run's full candidate ranking.

## Lightweight CLI smoke

When a search pilot produces no eligible failure rows, use the one-candidate historical
compatibility fixture at `tests/fixtures/adversarial_replay_gallery/issue_1501_compat/manifest.json`
to exercise schema-backed candidate selection, the real runner, and renderer. It is adapted from
the tracked #1501 `failure_0002` case and the #9645 replay; it is not a persisted search run or a
new discovery. Its complete `scenario_cert.v1` static-route certificate was generated post-hoc
from the tracked scenario by `scripts/tools/certify_scenarios.py`; the command, code revision,
input digests, and certificate digest are in
`tests/fixtures/adversarial_replay_gallery/issue_1501_compat/scenario_certification_provenance.json`.
This source-backed static classification lets the fixture pass the current strict selection gate;
it does not establish dynamic task feasibility, which remains unknown. Run it with a fresh output
path:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/tools/materialize_adversarial_replay_gallery.py \
  tests/fixtures/adversarial_replay_gallery/issue_1501_compat/manifest.json \
  --out output/adversarial-replay-gallery/<smoke-name> --top-k 1
```

The fixture's certificate is a post-hoc static route result, not an original search-time receipt or
a dynamic-feasibility oracle. The smoke should select and replay one case; compare the recorded
outcome and objective, then inspect `gallery_manifest.json`, the case manifest, and `figures/`.
Dynamic task feasibility remains unknown. A current-code replay at a different source revision is
an outcome reproduction only; video may be recorded as unavailable when the canonical runner emits
no video. The output stays in ignored `output/`; retain only a compact checksummed receipt when a
durable handoff is needed.

## Selection and replay checks

The selector requires an analysis-eligible row and a passed `scenario_cert.v1` receipt containing
exactly one certificate that validates against the canonical schema, matches the candidate
scenario's name/id, and has complete route accounting (`checks.route_count` equals the non-empty
`route_certificates` list). Top-level eligibility, route eligibility, and the all-routes check must
agree with their classifications; a `valid` or `hard_but_solvable` classification is admissible
only when every route is benchmark-eligible. Missing, failed, malformed, incomplete, ambiguous, or
scenario-mismatched certificates remain in candidate accounting and do not establish admissibility.
The selector also requires a finite objective, one unambiguous source episode, a
one-scenario YAML input, matching scenario and seed identity, candidate parameters matching the
generated scenario metadata, and a recomputed effective-scenario hash matching the search manifest.
The source failure attribution must agree
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
