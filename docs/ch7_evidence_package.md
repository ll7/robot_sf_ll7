# Chapter 7 evidence package

The release-level Chapter 7 package is built from existing, digest-verified inputs. It is a
cell-level descriptive package; it does not run a simulator, copy raw traces, populate the trusted
source registry, or admit dissertation evidence.

```bash
uv run python scripts/analysis/build_ch7_evidence_package.py \
  --source-package <approved-#6412-package> \
  --release-archive <release-0.0.3-archive> \
  --issue6814-compact <external-#6814-compact-packet> \
  --portfolio-config configs/analysis/ch7_worked_example_portfolio.v1.yaml \
  --config configs/analysis/ch7_evidence_package.v1.yaml \
  --output <package> \
  --check-determinism
```

The builder verifies the approved #6412 `SHA256SUMS` digest
`011c644bac469a1ce6255ddb8731c53c84bd310887759174f4c734b54d6bb543` and the release archive
digest `3cfefaaa39aab6cae541cece9573848a7e0afc5e1d9e4c9a7bbf48df2330b1a7`. The package retains the
complete 90-requested/88-admitted/two-excluded mapping ledger, a 672-cell audit atlas, a reduced
Chapter 7 atlas, release-cell context keyed by arm/configuration identity, deterministic Matplotlib
PDF/SVG output, sidecars, figure QA, and `SHA256SUMS`.

Before reading release rows, the builder validates the complete `issue_6814_compact_packet.v1`
schema, every compact checksum entry, the approved #6412 source digest, and the frozen portfolio
selection contract. The source mapping and release archive must agree on release digest/tag, and
selected release cells must reproduce their declared episode counts. A changed portfolio, forged
compact packet, incomplete source ledger, or incomplete selected cell stops the build.

The materialization overlay deliberately records `planner_upset` and `seed_sensitivity` as
unavailable. The #6814 doorway and seed-118 trajectory dossiers remain unavailable because their
starts/provenance are not compatible and `shared_prefix=false`; the package therefore emits viewer
and trace-publication unavailable receipts. The release-cell figures are preview artifacts pending
domain approval and are not a substitute for the exact-source admission gate.

The package must not pool hybrid arms as independent replications, infer a mechanism from a terminal
outcome, use DTW, introduce counterfactual branches, or claim causal divergence. A digest mismatch,
non-deterministic rebuild, missing report, or unsafe archive member stops the build.
