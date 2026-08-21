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

The reduced publication atlas uses `ch7-reduced-publication-atlas.v2` with the metric contract
`collision_count_mean_fields_are_per_episode_counts.v1`. Collision-related release-cell means are
emitted as `collision_count_mean`, `ped_collision_count_mean`, `obstacle_collision_count_mean`, and
`total_collision_count_mean`; they are count means from the source rows and may exceed 1. The builder
does not rename or rescale those source values into `*_fraction` fields.

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

## Terminal outcome labels

The reduced atlas `terminal_counts` field uses a stable, normalized label vocabulary. It is not a
verbatim count of the source episode `termination_reason` values. The builder applies this
precedence for each episode in [`_terminal_label()`](../scripts/analysis/build_ch7_evidence_package.py#L299-L310):

| Normalized label | Source condition |
| --- | --- |
| `route_complete` | `outcome.route_complete` passes the builder's boolean predicate; this takes precedence over later checks. |
| `collision_event` | `outcome.collision_event` passes the builder's boolean predicate and the route is not complete. |
| `timeout` | `termination_reason` is one of `terminated`, `timeout`, `max_steps`, `truncated`, or `horizon`, after the preceding checks. |
| `unavailable` | No recognized route, collision, or termination condition is present. |

Consequently, a `timeout` count can include source episodes whose raw reason is `terminated`.
The label describes the package's normalized terminal category, not a claim that every episode
exceeded a wall-clock or step limit. Consumers that need the raw reason must use the source episode
records; the frozen #6792 payload does not contain a raw-label breakdown.

Packages built by the current builder carry this mapping inside the package itself, so a consumer
can quote `terminal_counts` without reading this file. Both `manifest.json` and
`publication/reduced_atlas.json` emit a `terminal_label_normalization` block with contract
`ch7-terminal-label-normalization.v1`: the label precedence, the per-label source condition, the
`normalized_timeout_reasons` list (which always contains `terminated`), and
`raw_termination_reason_included: false`. The block is generated from the same constants the
builder applies, so the published statement cannot drift from `_terminal_label()`. It is required
by `ch7-reduced-publication-atlas.v2` and optional in `ch7-evidence-package.v1` only so that
packages built before it — including the frozen #6792 payload, which is not rebuilt — still
validate.

For a future package version, retain the normalized `terminal_counts` field for compatibility and
add an explicitly named per-cell raw-label breakdown (for example, `raw_termination_counts`) with
its own schema and source-provenance contract. That v2 change requires a new package digest and
reviewed admission decision; it is not a rewrite of the frozen #6792 payload.

## Author admission boundary

The builder's `blocked_pending_domain_approval` manifest is intentionally immutable. After the
author approves the exact package digest on [RobotSF issue #6792](https://github.com/ll7/robot_sf_ll7/issues/6792),
create a separate `ch7-evidence-admission.v1` receipt and populate the trusted source registry with
the approved source entry. The receipt binds the package `SHA256SUMS`, manifest, source package,
release archive, #6814 compact packet, registry, approval comment, role grains, unavailable reasons,
and forbidden claim classes. It is not valid unless all external inputs are supplied and rehashed:

```bash
uv run python scripts/analysis/verify_ch7_evidence_admission.py \
  --package-dir <package> \
  --source-registry configs/analysis/source_gate_registry.v1.json \
  --receipt <external-admission-receipt.json> \
  --source-package <approved-#6412-package> \
  --release-archive <release-0.0.3-archive> \
  --compact-dir <external-#6814-compact-packet>
```

The verifier rejects unlisted package or compact files, unbound or missing review sidecars, forged
approval IDs, digest mismatches (including source-package members and `package_complete.json`),
changed role scope, custom schemas, non-canonical registries, special filesystem entries, and any
package manifest that was rewritten to look admitted. The package payload must remain
byte-identical; the receipt is the effective admission record and remains outside the 21-file
package payload. The registry and receipt are an offline author-controlled trust anchor: the
verifier checks their exact digests and canonical issue-comment URL, but does not claim to
authenticate GitHub identity without a separate online review.

## Additive issue #7087 v2 projection

The v2 builder projects the immutable v1 audit into a new, blocked package
contract. It retains the existing 14-cell narrow-doorway terminal panel and
adds the requested 10-cell cross-topology and 4-cell cross-mechanism panels.
The in-repository `socnav_sampling` adapter is not asserted to be upstream
SocNavBench-equivalent.

```bash
uv run python scripts/analysis/build_ch7_evidence_package_v2.py \
  --source-package docs/context/evidence/issue_6792_ch7_evidence_package_v1 \
  --config configs/analysis/ch7_evidence_package.v2.yaml \
  --output output/ch7_evidence_package_v2 \
  --check-determinism
```

The v2 package emits only success fraction, near-miss mean, normalized
time-to-goal mean, and path-efficiency mean. Collision counts, collision
fractions, collision-derived composites, and SNQI (which is collision-derived
for this boundary) remain excluded under the issue #7042 ruling. The categorical
`collision_event` terminal label remains allowed in the retained terminal
panel and is not a collision-rate metric.

Every v2 cell binds to the v1 package checksum, source-member checksum, and
source-row checksum. The v2 manifest publishes the `terminated`-to-`timeout`
terminal-label mapping and `SHA256SUMS` covers the generated payload. The v2
package remains `not_admitted`; a maintainer-owned
`ch7-evidence-admission.v2` receipt is required before paper-facing use.

### Outcome-free admission diagnostic

Before domain review and durable source retrieval are available, validate a
generated package without creating or accepting a receipt:

```bash
uv run python scripts/analysis/verify_ch7_evidence_admission_v2.py \
  --package <package> \
  --check-only
```

The command verifies the blocked manifest schema and payload checksums, reports
the unresolved domain and receipt gates plus the frozen #7042 exclusion
boundary, and prints a
receipt-shaped template with unresolved approval, source-registry, and
retrieval fields. The template is explicitly `not_a_receipt` and is rejected
by the admission schema; no `admission/receipt.json`, empirical outcome, or
promotion decision is written. Run the command with `--receipt <receipt>` only
after a maintainer-owned receipt exists and the package has independently
crossed the domain-approval and durable-receipt gates.

The diagnostic accepts either fresh builder output without review sidecars or
a durable package with a complete, byte-preserving review-sidecar set. Partial
or unbound sidecars fail closed. The #7042 exclusion remains a frozen metric
boundary; it does not itself admit the package or authorize paper-facing use.

### Deterministic build receipt (#7410)

The durable build receipt
[`issue_7410_ch7_evidence_build_receipt.v1.json`](context/evidence/issue_7410_ch7_evidence_build_receipt.v1.json)
records the exact v2 source bindings, build commit/tree, builder and verifier hashes, Python/uv
and lockfile identity, two independent output-tree hashes, package manifest/checksum hashes, and
the successful outcome-free `--check-only` result. It is adjacent to the v2 package because the
package and its `SHA256SUMS` are immutable; review-only `.review.json` sidecars are excluded from
the recorded payload-tree hash.

Generate it only after the canonical builder code and schema are committed, then verify it from a
clean checkout with:

```bash
uv run python scripts/analysis/build_ch7_evidence_build_receipt_v1.py \
  --receipt docs/context/evidence/issue_7410_ch7_evidence_build_receipt.v1.json \
  --check-only
```

The receipt's self-hash covers canonical JSON with only `#/integrity` excluded. This is build
provenance, not an admission receipt, domain approval, publication authorization, benchmark result,
or paper-facing evidence. The separate `ch7-evidence-admission.v2` receipt remains required.
