# Issue #9656: historical benchmark hard-case mining

[Back to the context index](INDEX.md) · [Issue #9656](https://github.com/ll7/robot_sf_ll7/issues/9656)

## Result

The checksum-pinned Benchmark Release 0.0.2 source supports a diverse slice of 36 stable cases:
seven planners, six scenario families, and seven scenario IDs. The selector takes 15 cases from
each of nine named event/metric groups, with overlap reducing the union to 36 unique case IDs.
Materialization writes a one-row scenario matrix with the original seed and a planner configuration
snapshot for every selected case.

The source contains 241 canonical collision events with collision termination. For every such row,
both `metrics.collisions` and `metrics.total_collision_count` are available and non-positive. The
slice preserves the event and metrics independently and flags 17 selected cases with that
contradiction; it does not repair the source values. The camera-ready analyzer's seven
planner-level integrity findings remain visible in the evidence packet.

Four distinct cases each produced one episode row on the corrected replay run. Their scenario,
seed, planner/config identity, and all three canonical event flags matched the selected source
rows, but named metrics differed in every case. The replay checkout
(`5cccee50be333adceee4c978b54bf63d32454cc9`) differs from the source campaign
(`f7ebdcae2375d085e925213197a75a386e26a79c`), so every result is
`mismatch_different_revision`, not an exact historical replay.

## Source and reproducibility

The source archive is the public
[Benchmark Release 0.0.2 bundle](https://github.com/ll7/robot_sf_ll7/releases/download/0.0.2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz),
SHA-256 `64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90`. Its campaign has
987/987 expected episode identities present, with no missing, duplicate, or malformed rows. The
exact source matrix is `configs/scenarios/classic_interactions_francis2023.yaml`, SHA-256
`d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5`.

Case selection is owned by the merged `benchmark-showcase.v1` capability from PR #9662. This packet
labels the upstream campaign as `source_campaign_id` so the mined slice is not represented as a
second campaign. It preserves the selector's original execution revision
`477f14c1b4b052ad407a34a71caace6618a75eeb`
and summary SHA-256 `c2f0b4c0b85303e3547e4ce13f5676b45c886b6b9593a78a7a4d6016fdb39c1f`. The three
selector source-file SHA-256 values are retained in the evidence summary and match the reachable
snapshot at `0a5f73283b98279900797b75d20adf6d4676086b`; this verifies matching source files without
rewriting the historical execution revision. Repeated selection at the same output path produced
the same summary hash and the same case IDs at another output path. The full analyzer subreport
records absolute output paths, so running in a different directory changes that subreport's hashes.
The materializer consumes the versioned JSON contract without copying or reranking the selector.

Reproduction commands and the machine-readable 36-case inventory are in the tracked
[evidence bundle](evidence/issue_9656_hard_case_mining_2026-09-24/payload/summary.json), with a
[human report](evidence/issue_9656_hard_case_mining_2026-09-24/payload/report.md),
[bundle manifest](evidence/issue_9656_hard_case_mining_2026-09-24/evidence_bundle_manifest.json),
and [checksums](evidence/issue_9656_hard_case_mining_2026-09-24/checksums.sha256). The original
release and all source episode rows remain the durable raw evidence; extracted payloads, case
JSON files, replay matrices/configs, and replay episode rows stay in ignored `output/` caches.
Source cases can be rematerialized from the release. The summary retains hashes and metrics for the
four replay rows, but not their raw bytes; verify those exact receipts only while the local cache is
preserved. A later replay is a new evaluation, not a reconstruction of those bytes. On
`--resume-from`, replay identity, execution mode, planner config, events, and metrics are recomputed
from the copied episode row against its checksum-pinned source row. Only matching, resolved
`native`, `adapter`, or `mixed` execution modes are eligible; `unknown` does not establish an exact
replay. If the prior receipt did not contain an episode-output checksum, an otherwise exact
comparison is retained as `replay_artifact_checksum_unverified`; capturing the current file hash
establishes custody from that resume onward, not integrity of the original output at run time.
An `exact_match` also requires an eligible source case, successful and available source/replay
execution metadata, no nested fallback or degraded marker (including positive fallback counters),
no canonical invalid-run state on either row, matching non-empty planner config hashes, and clean,
unchanged checkout snapshots immediately before and after replay at the recorded revision.
The scenario map and each configured model/checkpoint input must also resolve to a regular file in
the recorded Git tree, and the bytes visible to replay must match that tree entry. External,
ignored, untracked, transformed, missing, or otherwise unverified runtime inputs leave exact replay
identity unavailable.
Nested runtime status values must be recognized available statuses; unknown or malformed values
remain unavailable. Explicit `unavailable` markers must be booleans, and only literal `false` is
accepted as available evidence; `true` and malformed values remain unavailable. Each
`replay-checkout-snapshot.v1` snapshot must include its status-entry list,
agree that `clean` is true exactly when that list is empty, and bind the list to its recorded
porcelain SHA-256 digest. A dirty checkout, invalid-run row, unknown runtime status, moved `HEAD`,
changed working-tree status digest, or unavailable/malformed snapshot blocks exact-match
classification. On resume, the manifest replay receipt must also match its per-case receipt before
an otherwise exact replay can remain `exact_match`.
A resumed attempt whose replay directory is missing remains `attempted` and is counted as
`replay_artifact_missing_on_resume`; the missing artifact does not reset it to `not_attempted`.
Legacy receipts without both checkout
snapshots remain `replay_checkout_cleanliness_unavailable` when their rows otherwise match, even
when they contain a clean pre-run flag; post-run checkout evidence is never inferred on resume.

## Replay boundary and failed attempt

An initial attempt used `robot_sf_bench run --scenario-id`. The runner scheduled three seed jobs
per attempted case but resolved the map from `scenario_path='.'` and produced zero episode rows
(12 failed setup jobs across the four distinct cases). The implementation repaired this by
materializing an explicit one-row matrix with the canonical map search path and exactly the source
seed, then invoking the runner without `--scenario-id`. Those four corrected invocations produced
one episode row each. The failure attempts, exact commands, return codes, and log hashes are
retained separately in the summary; they are not planner outcomes. Four distinct case IDs were
touched, below the issue's five-case replay ceiling.

Five selected PPO rows remain unavailable because their archived model files are missing; 27
selected cases were not attempted. The release contains no `replay_steps`, so the showcase renderer
reports all 36 source trajectories as unavailable. No trajectory/video renderer was called.
Replay episode checksums were first captured while resuming existing output because those original
receipts lacked output hashes; later copies matched. The source-row comparison is rederived on
resume, and no successful comparison without its prior checksum is promoted to an exact match. This
records local artifact custody from the hash capture onward, not a signed checksum emitted by the
original runner process. The historical release does not record its execution environment; replay
Python/platform and `uv.lock` digest are included in the summary.

## Claim boundary and ownership

This is diagnostic evidence from one historical simulator release and four bounded reruns. It is
not a planner ranking, generalization result, safety claim, or real-world safety statement. It does
not admit cases into issue #9652's versioned corpus and does not build visualizations owned by
Issue #9647. Cases remain regression candidates pending the corpus owner's admission rules.

The selector producer is the merged capability from PR #9662; this implementation consumes its
versioned summary and does not duplicate case ranking. The derived packet labels the upstream
identifier `source_campaign_id`, so it does not register a second campaign. The source
collision/count inconsistency is retained as a data-quality limitation rather than a reason to infer
or rewrite the canonical event.
