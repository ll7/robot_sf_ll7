# Adversarial counterexample corpus

Current contract: [issue #9652](https://github.com/ll7/robot_sf_ll7/issues/9652). The
versioned corpus API and CLI are the implementation surface for the planned #9653 loop; the
initial fixture is the persisted [#9645 bounded pilot](https://github.com/ll7/robot_sf_ll7/issues/9645)
and its separately replay-verified #1501 case.

The versioned corpus in `robot_sf.adversarial.counterexample_corpus` stores
admitted challenge cases, search-run provenance, admission attempts, and planner
observations. A case remains in the corpus after another planner solves it.
Planner status is computed from the retained observations for a selected planner
and configuration; no stored discovery or failure record is rewritten.

## Initialize and import the bounded pilot

The initial #9645 packet contains a zero-discovery search result and a separate
historical #1501 collision that was replayed twice under one current revision.
The importer records the pilot's 64 completed candidates and explicit zero
discoveries, then applies the admission checks to the independently replayed
historical case. It does not treat the pilot's successful candidates as newly
discovered counterexamples.

```bash
uv run python scripts/tools/manage_adversarial_counterexample_corpus.py init \
  --corpus output/adversarial-corpus/corpus.json

uv run python scripts/tools/manage_adversarial_counterexample_corpus.py import-9645 \
  --payload docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload \
  --corpus output/adversarial-corpus/corpus.json \
  --corpus-root output/adversarial-corpus
```

Rejected admission attempts are persisted in the corpus and cause the import
command to exit nonzero. The importer binds both current replay JSONL files,
their provenance, the scenario and route inputs, the archived source record,
the historical search source snapshots, and the #9645 accounting packet. It
checks the packet's outer file inventory and checksum sidecar, compares the
source-hash receipt with run metadata, and verifies each consumed payload file
before admission. The corpus retains the outer manifest and checksum sidecar
digests alongside the copied accounting evidence.

The #1501 raw historical episode and original search manifest were not
archived. The case therefore records the historical source revision as lineage,
the manifest as unavailable, and the historical raw replay match as
unverifiable. Its regenerated target and replay revisions match exactly; the
selected event and metric projections agree across both runs and with the
durable receipt. The persisted #9645 packet rewrites local paths in its bundled
replay artifacts, so each replay receipt keeps the source digest before path
normalization separate from the normalized bundle digest. The normalization
receipt pins the allowed local-path rewrites.

Dynamic feasibility for this case remains `admissible_feasibility_unknown`.
The current `scenario_cert.v1` result is a static route certificate, not proof
that the dynamic task is feasible. The #9656 mined receipts are not admitted by
this importer: their replay revision differs from the source campaign revision
and they remain candidates until exact replay evidence satisfies the same
revision and input checks.

## Record an evaluation and recompute status

An evaluation JSON object can be appended with `record-evaluation`. Complete
observations require canonical outcome flags, a replay episode digest, metrics,
planner/config identity, and explicit execution/readiness/availability state.
Failed, partial, missing, and unknown observations can also be retained with an
explicit `evidence_status` and a reason; they contribute `unknown` status rather
than being dropped or counted as a solve. Fallback or degraded complete rows
remain visible and also cannot count as a solve.

```bash
uv run python scripts/tools/manage_adversarial_counterexample_corpus.py record-evaluation \
  --corpus output/adversarial-corpus/corpus.json \
  --observation evaluation.json

uv run python scripts/tools/manage_adversarial_counterexample_corpus.py status \
  --corpus output/adversarial-corpus/corpus.json \
  --planner-id goal \
  --planner-config-identity 44136fa355b3678a
```

Status is specific to the exact planner ID and configuration identity:

- `solved`: every complete, eligible observation for the pair succeeds and no
  incomplete, fallback, degraded, or contradictory row is present;
- `unsolved`: every complete, eligible observation has the challenge outcome;
- `mixed`: complete eligible observations disagree;
- `unknown`: no complete eligible observation exists, or any observation is
  incomplete, degraded, fallback, unavailable, or contradictory.

## Export a replay slice

The slice contains a canonical scenario matrix, copied route overrides, a
planner configuration snapshot per case, a checksum manifest, and a replay
command for each case. Export refuses to overwrite an existing directory and
checks the stored scenario/route bytes against their recorded digests and
effective scenario identity.

```bash
uv run python scripts/tools/manage_adversarial_counterexample_corpus.py export-slice \
  --corpus output/adversarial-corpus/corpus.json \
  --corpus-root output/adversarial-corpus \
  --output-dir output/adversarial-corpus/regression-slice
```

From the exported directory, use a listed command such as:

```bash
uv run robot_sf_bench run --matrix replay_matrix.yaml \
  --out results/<case-id>.jsonl --algo goal \
  --algo-config planner_configs/<case-id>.yaml \
  --scenario-id crossing_ttc_template_adversarial_0008 --no-video
```

The slice is a set of replay inputs, not evidence that a new execution matches
the admission receipt. Each new result must be checked and appended as another
planner observation. The corpus does not provide a mathematical feasibility
oracle, merge near-duplicates, or establish search-space coverage or real-world
safety.
