# Guard and escape ablation evidence

**Diagnostic-only, development seeds 1001–1030.** This is a new execution of
one controller on a fixed 18-cell roster, not historical release evidence or
an explanation of the ranking among different planners. All 2,160 episodes
declare the same differential-drive adapter; fallback and degraded counts are
zero. The independent review is **CONFIRMED**. Packaging supplies provenance,
not manuscript admission or a newly registered decision rule.

| Group | Guard success effect, pp | Pointwise 95% interval, pp |
| --- | ---: | ---: |
| Static constriction (8 cells) | +46.0 | [+42.5, +50.0] |
| Portal (7 cells) | +34.0 | [+30.7, +37.1] |
| Crowding control (3 cells) | −1.1 | [−6.7, +5.0] |

Unrounded values, escape effects and interactions are in
[`evidence/summary.json`](evidence/summary.json) and all four endpoints in
[`evidence/paired_effects.csv`](evidence/paired_effects.csv). Both CSVs reproduce
byte for byte from the compact episode inputs using the portable analyzer and
the archived original analyzer. Intervals condition on this fixed roster;
they are not simultaneous intervals or uncertainty over unseen scenarios.

## Design and interpretation

The study crosses guard on/off and escape on/off within the continuous
`hybrid_rule_v4_fast_progress_static_escape_continuous` controller: four arms,
18 cells, 30 paired development seeds, 540 episodes per arm. Group membership,
authored budgets, exact switches, source commits and analysis are frozen in
[`inputs/study.yaml`](inputs/study.yaml). The equal-cell main effects average
on-minus-off over the other factor. Ten thousand bootstrap draws resample the
30 complete seed blocks, retaining every cell and arm; NumPy `default_rng(1001)`
and the 2.5th/97.5th percentiles match the original analysis.

The guard intervention removes the specified collision, clearance and braking
rejection bundle while retaining speed limits, prediction and soft scoring.
Escape-off disables four named switches, not every route/recovery behavior.
Some switched guard branches are inactive under these configs, so the bundle
effect cannot identify individual components. The guard-off wall collisions
do not establish a separately identified mediation effect, a distinctive hybrid
advantage, or an explanation of historical rankings. Portal is a confinement
companion; no historical hybrid lead there is assumed. Escape's confinement
main effects fall below the supplied 5 pp rule; a small increment with the guard
on does not establish equivalence or prove that recovery has no role.

The original contract does not register the 5 pp/RQ-count rule or define MIXED.
The review applies the separately supplied rule but cannot authenticate its
pre-result wording/date. This documentary limit does not change the numerical
verification. See the complete
[`independent review`](provenance/independent_review.md).

## Portable reproduction

From the repository root, with Python 3.13.14, NumPy 2.4.6 and PyYAML installed:

```bash
uv run python docs/context/guard_escape_ablation/analyze.py \
  --output /tmp/guard-escape-recomputed --verify
sha256sum -c docs/context/guard_escape_ablation/SHA256SUMS
```

`--verify` requires the generated 48 rates, 36 contrasts and summary to match
the frozen files byte for byte, and separately checks all three headline
values at one decimal place. Output must be outside the immutable packet.
It needs no simulation, raw archive, network, model weights, or private path.
For a sibling checkout using an existing project environment, use the repository's
`scripts/dev/run_worktree_shared_venv.sh --standalone --venv <environment> --`
prefix before `uv run python ...`.

## Packet and custody

- [`inputs/`](inputs/) holds the scientific spec, four job definitions, arm
  configurations and 18-cell scenario matrix. These are custody copies; the
  campaign config is outside the active benchmark namespace. Paths in the
  campaign/scenario copies are made repository-relative; all source digests
  and transformations are recorded.
- [`evidence/episodes.jsonl.gz`](evidence/episodes.jsonl.gz) retains one row per
  episode: factor/seed/cell identity, exact endpoint metrics/outcomes, execution
  validity, budget/termination, source-file path, line number and SHA-256 of the
  exact producer line (including its newline). It omits trajectories and raw
  diagnostics. [`project.py`](project.py) regenerates it from the four original
  producer files using the pinned runtime's canonical fallback parser.
- [`provenance/producer_files.json`](provenance/producer_files.json) binds the
  projection to four complete producer-file digests, also independently recorded
  in [`independent-results.json`](provenance/independent-results.json).
- [`provenance/custody.json`](provenance/custody.json) binds all copied/derived
  artifacts to their original digests. Private paths in the review and selected
  receipts are replaced by symbolic roots; the original hashes are retained.
  The original bundle's 349 payloads were rechecked against its
  [`manifest`](provenance/original_bundle_MANIFEST.sha256), whose digest is
  `22ab003db1e2dd138e2830ca088d88fe45296f7fa2f4934b7fd4e7cbebf7316d`.
  The 350th file is that manifest itself. Earlier two-copy cold-readback
  receipts are preserved as metadata; no fresh remote readback is claimed.
- The approximately 400 MB full bundle remains in its pre-existing two-copy
  archive. It is not added to Git, consistent with
  [`benchmark_artifact_publication.md`](../../benchmark_artifact_publication.md)
  and the `artifact-provenance` compact-evidence rule. The task handoff retains
  its private retrieval locators. The tracked compact copy suffices to reproduce
  every reported number. W&B upload was pending in the source receipts.

The acquisition ran at `0035fba05c9b679e39f58ec286f5d57b96bb9a0a`, based on
public release `373dbfde4f39667cf9e8732dabe7df5118bdeab1`. The complete
[`acquisition-source.patch.gz`](provenance/acquisition-source.patch.gz) decompresses to a patch that reconstructs
the two changed source/test files from that release; the narrower original
runtime patch and byte-exact original analyzer are also retained. Historical
behavior/config validation receipts are supporting source evidence, not new
simulation checks. Full acquisition replay requires that recorded runtime and
environment; the job command in `inputs/jobs.json` is documentary and was not run.

This packet changes no runtime, active study config, frozen artifact, evidence
admission state, dissertation prose or claim. Reopen if a source digest changes,
pairing or factor binding fails, or the original preregistration establishes a
different decision rule.
