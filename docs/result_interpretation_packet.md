# Result Interpretation Packet (`result_interpretation_packet.v1`)

**Contract version:** `result_interpretation_packet.v1`
**Schema:** `robot_sf/benchmark/schemas/result_interpretation_packet.v1.json`
**Module:** `robot_sf/benchmark/result_interpretation_packet.py`
**Script:** `scripts/analysis/build_result_interpretation_packet.py`
**Issue:** #7029

## Purpose

The result interpretation packet is a **contract-only** container for structured
interpretation of benchmark or diagnostic results.  It answers three questions:

1. **What question does the evidence address?**
2. **How was it computed, and what population/modes does it cover?**
3. **What is it allowed and forbidden to claim?**

Each packet also records an explicit controlled evidence tier and admission state.
This keeps a nominal benchmark result, diagnostic observation, and visualization
fixture distinct even when they share the same packet schema.

Packets may carry an optional top-level `claim_identity` object with the exact
`campaign_id`, `question`, and `estimand` supported by the packet. A
decision-capable answerability proof requires this object and compares it with
the manifest; a packet's question and estimand fields must match it as well.
Older diagnostic and fixture packets may omit the field.

The packet does not re-run experiments or infer values from filenames or plots.
It preserves existing `artifact_catalog` and `figure_qa` contracts by referencing
artifacts via `file_ref` digests rather than re-registering them.

## Claim boundary

Every packet carries an explicit `claim_boundary` with `allowed` and `forbidden`
lists.  The packet is the *minimal* interpretation surface: it states what the
evidence supports and what it explicitly does not.

### Allowed claims (examples)

- Bounded simulator-defined metric-family effects on a frozen scenario surface.
- Diagnostic-only observations without inferential comparison.
- Visualization-only process contrasts with causal abstention.

### Forbidden claims (examples)

- Planner ranking or universal superiority.
- Causal mechanism or population generalization.
- Paper-facing or dissertation-ready claims.
- Real-world, fairness, deployment-ethics, or welfare claims.

## Decision vocabulary

All decisions use exactly one of five controlled outcomes:

| Outcome | Meaning |
|---------|---------|
| `supported` | The evidence reaches the preregistered support threshold. |
| `not_supported` | The evidence does not reach the threshold; reason recorded. |
| `inconclusive` | The evidence is insufficient to decide. |
| `invalid` | The source or execution is invalid; excluded from evidence. |
| `unavailable` | The metric or comparison is not available for this packet. |

## Fail-closed validation

The validator rejects at minimum:

- Missing denominator or analysis unit.
- Unsupported zero imputation (`not_imputed` missingness).
- Undefined comparator direction.
- Inferential comparison without uncertainty declaration.
- Unrecorded multiplicity.
- Source/packet/figure/caption/review digest drift after review.
- Caption assertion text/status mismatch or `inferred` status.
- Caption assertions without controlled `bound_to_packet_fields` references.
- Caption-file bytes that differ from the deterministic `render_caption(packet)` output.
- Forbidden claim escalation via decision outcome.
- Duplicate IDs across metrics, decisions, figures, and sources.
- Supported decisions without a decision-level `contrast_result` binding their
  comparator, effect, support denominator/threshold, null, uncertainty, and
  multiplicity.
- Unsupported decision vocabulary.
- `support > denominator` violations.
- Supported outcomes without a directed comparator, finite effect, declared
  support threshold, or complete native/adapter-only execution population.
- Population accounting errors (`included + excluded != total`).
- Missing, duplicate, or under-counted rejected-row ledger entries.
- Empty claim boundary lists.

Evidence tiers are controlled (`smoke_diagnostic`, `visualization_fixture`,
`nominal_benchmark`, or `paper_grade`) and admission states are controlled and
cross-checked. `admitted` or `paper_grade` packets require an independent
reviewer, a reviewer commit, and both exact review digests. A producer cannot
serve as its own reviewer.

Caption assertions carry a controlled template ID. Available figure links must
also carry a checksum-bound `artifact_catalog` reference; the validator loads
the existing catalog, checks the catalog commit, output identity, caption
identity, PNG/PDF/SVG signature, and exact caption bytes against the packet's
controlled renderer. An unavailable figure remains explicit and does not require
rendered bytes. This packet contract does not replace or re-register the
repository's artifact catalog.

Source references are repository-relative durable files.  Each source must carry
its SHA-256 digest, generation commit, tracked commit, and generation command.
The generation commit records how the source was produced; `tracked_commit`
binds the digest to the exact repository bytes that are retained.  Validation
hashes both the current file and the tracked bytes and fails closed on drift,
missing files, unavailable commits, or commands that do not name a script
present at the generation commit.  Review-marker sources use the explicit
`evidence-review-marker.v1` command.  Rendered figure links are similarly
required to resolve to durable repository files with matching digests.
Execution-mode counts must reconcile exactly with the included population.  A reviewed packet
must bind both `reviewed_packet_digest` and `post_review_digest`; the latter
includes the reviewer identity and cannot be supplied as an arbitrary hex value.
Every metric also names one or more declared `source_ids`, so a reviewer can
start from the metric's retained source or aggregate artifact when reproducing
its support, denominator, effect, uncertainty, and null values. Unknown or empty
metric source bindings fail closed.

For a `supported` pairwise decision, the validator goes further: exactly one
registered `report_summary` source must contain the matching metric and planner
pair. The packet's support, denominator, effect, confidence interval, raw
p-value, and adjusted p-value must match that source row (within the documented
display-rounding tolerance), and the source row plus packet must satisfy the
source's declared adjusted-alpha decision rule. The packet's `report_summary`
source reference must explicitly declare the paired effect convention
`comparison_minus_reference`, and the packet direction must bind to that
declaration as well. A structurally valid or statistically favorable number
cannot claim `supported` without this source-row binding.
For supported decisions, the metric-level support, denominator, and threshold
must also agree with the decision-specific structured contrast; a packet cannot
hide inconsistent accounting in a shared metric summary.
Supported packets must also register exactly one machine-readable
`preregistration` source. Its result contract binds the metric unit and
denominator, null value, uncertainty and multiplicity declarations, comparator
direction, and support threshold. For the full paired campaign contract, the
threshold is derived from the preregistered seed and scenario dimensions rather
than accepted as a packet-authored number. The report summary's validation block
must agree with the packet's population and execution-mode counts, and the
shared metric effect and uncertainty must bind to the packet-level primary
comparator row. The packet-level comparator used by observed captions must
match one of the supported decision comparators.
Packets also list `fail_closed_changes`, making the retained exclusions and
claim refusals explicit in the review report.  The top-level `forbidden_claims`
mirror the claim-boundary refusal list, and a small high-risk phrase guard
prevents supported decisions, allowed claims, findings, or observed captions
from silently escalating beyond that boundary.

Supported decisions additionally require complete metric data, a declared null
value, at least one observed interval or p-value, and no fallback/degraded rows.
Each supported pairwise decision must carry its own structured `contrast_result`
with comparator-matched effect, support accounting, null value, uncertainty, and
multiplicity; shared metric summaries cannot substitute for decision-specific
statistics. Any supplied `contrast_result` is validated even when the outcome is
`not_supported`, `inconclusive`, `invalid`, or `unavailable`. Population attrition
records both `invalid` and `rejected` rows explicitly, and every rejected row must
have a unique `row_id` and declared reason in the complete `rejected_rows` ledger.
Checksum manifests include generated outputs plus every source and available
rendered-figure/caption file referenced by the packet.

## Source fixtures

Three compact, deterministic fixtures reference tracked repository evidence:

| Fixture | Source issue | Decision |
|---------|-------------|----------|
| `issue_6474_comfort_exposure_supported.json` | #6474 | `supported` |
| `issue_6944_brne_candidate_transition_diagnostic.json` | #6944 | `not_supported` |
| `ch7_visualization_causal_abstention.json` | #6792 | `unavailable` |

Fixtures bind source paths and SHA-256 digests, contain structured caption
assertions generated from packet fields, and represent figure availability
explicitly without inventing rendered results. The #6474 fixture takes each
pairwise support, denominator, and raw p-value directly from its checksum-bound
report (`n_paired=180` per comfort-exposure contrast). Artifact-catalog test
entries use `source_kind: fixture_construction` and a `fixture-construction:`
provenance marker so committed literal fixture bytes are not misrepresented as
outputs of the catalog validator.

## Usage

### Validate a packet

```bash
uv run python scripts/analysis/build_result_interpretation_packet.py \
    --input packet.json --validate-only
```

### Build and write validated output

```bash
uv run python scripts/analysis/build_result_interpretation_packet.py \
    --input packet.json --output validated.json

# Optional deterministic caption, review report, and checksum manifest
uv run python scripts/analysis/build_result_interpretation_packet.py \
    --input packet.json --output validated.json \
    --caption-output caption.txt \
    --review-output review.json \
    --checksum-output SHA256SUMS
```

### Programmatic API

```python
from robot_sf.benchmark.result_interpretation_packet import (
    build_and_validate_packet,
    compute_packet_digest,
    load_result_interpretation_packet,
)

packet = load_result_interpretation_packet(Path("packet.json"))
digest = compute_packet_digest(packet)
```

## Relationship to existing contracts

- **`artifact_catalog.v1/v2`**: The packet references figures through a
  checksum- and commit-bound catalog entry without re-registering the artifact.
- **`figure_qa.py`**: Caption assertions in the packet are compatible with
  the v2 figure semantics QA surface.
- Figure links carry a visual contract bound to the packet estimand, with an
  explicit encoding rationale covering plot type, encodings, transforms,
  limits, reference lines, ordering, faceting, uncertainty, sample-size
  display, legend identities, and accessibility.
- **`benchmark_claim.v1`**: A packet may be a component of a broader claim
  bundle but does not establish a claim on its own.
- **`ch7_case_portfolio.v2`**: The Chapter 7 visualization fixture preserves
  the controlled narrative grammar and causal-abstention contract.
