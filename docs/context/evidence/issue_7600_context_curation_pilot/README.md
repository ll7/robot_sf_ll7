# Context-note Curation Pilot

This is the deterministic, fail-closed 50-note pilot for [Issue #7606](https://github.com/ll7/robot_sf_ll7/issues/7606),
under parent backlog [Issue #7600](https://github.com/ll7/robot_sf_ll7/issues/7600). It validates a
curation process; it does not establish that the remaining orphan notes are obsolete, safe to
move, or safe to omit from retrieval.

## Source and input

The source state is commit c71543025108710791b37cafea35b9c8fe3b391e. The canonical freshness
checker reported 498 tracked top-level docs/context/*.md notes absent from both INDEX.md and
catalog.yaml. The input digest is the SHA-256 of the sorted repository-relative orphan paths,
joined with one newline per path:

b24638ad5a61bd4da469e63313a098b8f805ad45123106e998e20035dd4699e7

The complete 50-row result is [inventory.json](inventory.json). For each row, inbound-link count
means the number of distinct tracked Markdown source files under docs/context/ that resolve to the
note using the same relative-link rules as the freshness checker.

## Deterministic selection

The sample is reproducible from the source state without hand selection or a random seed:

1. Generate the canonical orphan list and sort repository-relative paths for the input digest.
2. Sort rows by last committed touch timestamp, then path; select the ten oldest and ten newest.
3. Select the oldest and newest row in each detected family, using basename/title tokens rather
   than the generic docs/context/ directory name.
4. Select the earliest row for each available global cell of issue-specific/non-issue-specific
   and positive/zero inbound links.
5. Fill remaining slots with evenly spaced lexical paths, then lexical order, deduplicating and
   stopping at exactly 50 rows.

The pilot covers all seven detected families (benchmark, evidence, general, release, research,
training, and workflow), 25 rows with positive and 25 with zero inbound links, and 40
issue-specific and 10 non-issue-specific rows.

## Disposition contract and result

The canonical policy is documented in [the context workflow README](../../README.md). A filename
or date heuristic can propose a disposition, but finalization requires the structural evidence
and precedence rules there. The pilot stopped before automatic metadata changes because the
owner-decision rate exceeded the 20% stop threshold:

| Final disposition | Rows |
| --- | ---: |
| active_index | 0 |
| historical_keep | 2 |
| superseded | 0 |
| archive_candidate | 0 |
| needs_owner_decision | 48 |

The ambiguous rate is 96%. The 2 historical rows have explicit execution-history markers. The
other 48 rows remain needs_owner_decision, including all rows with evidence/release/legal/claim
sensitivity that lack the required owner decision. No pilot note was deleted or moved, and no
pilot note received an INDEX or catalog row. The two evidence artifacts themselves are registered
as evidence entries in catalog.yaml so the repository integrity contract can discover them. The
orphan count is 498 before and after the pilot.

Use owner-first follow-up batches of 10 notes (about 50 batches for the full orphan set). If the
pilot rate held, planning would require decisions for approximately 478 rows; this is a workload
estimate, not a statistical confidence interval. The pilot therefore recommends resolving
ambiguous evidence and provenance first before considering any archive moves or metadata additions.

## Reproduction and validation

~~~bash
uv run python scripts/tools/check_context_note_freshness.py \
  --index docs/context/INDEX.md \
  --context-dir docs/context \
  --catalog docs/context/catalog.yaml \
  --json-output /tmp/issue7600-orphans.json
bash scripts/dev/check_context_notes.sh
uv run python scripts/dev/check_docs_evidence_integrity.py --full
git diff --check
~~~

These checks validate the curation report and repository integrity only; they do not promote any
pilot row to benchmark, release, or scientific evidence.
