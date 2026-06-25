# Issue #3477 — explicit Ruff security (Bandit `S`) baseline

**Status:** security-governance increment. Converts the blanket `S` suppression into an
explicit scoped baseline so new categories of security finding are enforced.

## What changed

`pyproject.toml` previously ignored the entire Ruff `S` (Bandit) category. It now lists only
the **15 codes currently present in the tree**, so every *other* `S` rule (≈50: `exec`/`eval`
S102/S307, `yaml.load` S506, jinja autoescape S701, request-without-timeout S113, etc.) is now
**enforced** repo-wide. No existing finding had to be fixed — `ruff check .` stays green — and a
probe confirms a new `exec` (S102) now errors.

## Baselined (suppressed) codes — the ratchet-down backlog

`S101` assert · `S105` hardcoded-password (mostly false positives) · `S108` temp-file ·
`S110`/`S112` try-except-pass/continue (also tracked by the broad-exception ratchet) · `S202`
tarfile · `S301` pickle · `S310` url-open · `S311` non-crypto random · `S314` xml ·
`S324` insecure-hash · `S602`/`S603`/`S607` subprocess · `S608` SQL string.

These are the gradual-rollout backlog (the issue calls for an explicit baseline, then ratchet).
`.github/workflows/security-baseline.yml` continues to expose full Bandit findings.

## Guard

`tests/test_ruff_security_baseline.py` asserts the `S` category is never blanket-ignored again
(only specific codes) and that the explicit baseline codes remain listed until ratcheted down.

## Follow-ups (out of scope here)

Per-code triage to ratchet the backlog down — e.g. justify/scope the subprocess and pickle uses,
replace insecure hashes used for security, audit the S105 hits — each as a small reviewed change.
