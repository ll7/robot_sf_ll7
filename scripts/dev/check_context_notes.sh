#!/usr/bin/env bash
set -euo pipefail
# Canonical context-note integrity checks (issue #7401).
#
# The guessed scripts/dev/check_docs_context.py does not exist.
# Use this wrapper or the three explicit commands below:
#
#   1. Changed-doc / proof check (diff-scoped, includes catalog provenance):
#      uv run python scripts/validation/check_docs_proof_consistency.py --base "${BASE_REF:-origin/main}" --check-context-note-freshness --freshness-scope diff
#   2. Full evidence integrity (link + catalog/evidence coverage):
#      uv run python scripts/dev/check_docs_evidence_integrity.py --full
#   3. Freshness detail (strict vs non-strict, orphan warning set):
#      uv run python scripts/tools/check_context_note_freshness.py --index docs/context/INDEX.md --context-dir docs/context --catalog docs/context/catalog.yaml
#      (without --strict, the large pre-existing orphan-note warning set is non-blocking; --strict fails on it)
#
# Non-strict orphan warnings are repository backlog, not per-PR failures for a changed note.
BASE_REF="${BASE_REF:-origin/main}"
echo "1/3 changed-doc proof consistency (base=$BASE_REF) ..."
uv run python scripts/validation/check_docs_proof_consistency.py --base "$BASE_REF" --check-context-note-freshness --freshness-scope diff
echo "2/3 evidence integrity (--full) ..."
uv run python scripts/dev/check_docs_evidence_integrity.py --full
echo "3/3 context-note freshness (non-strict, catalog+index coverage) ..."
uv run python scripts/tools/check_context_note_freshness.py --index docs/context/INDEX.md --context-dir docs/context --catalog docs/context/catalog.yaml
echo "OK check_context_notes.sh done"
