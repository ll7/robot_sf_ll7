# Author Trust Gate

[Back to Documentation Index](../README.md)

Agent workflows read issue bodies, pull-request bodies, comments, and review
comments and place that text into prompts, capsules, and loop decisions. Only
content authored by our own operator account (`ll7`) is auto-ingested. Every
other author — including bots, GitHub Actions accounts, external contributors,
and deleted or missing authors — is untrusted by default. Our own user can
explicitly opt foreign content in with a flag, and the receipt records that
choice with attribution.

The canonical helper is `scripts/dev/agent_content_gate.py`. It is fail-closed:
when author metadata is missing or ambiguous, the content is never trusted.

## Trust policy

| Author | Classification | Auto-ingested |
| --- | --- | --- |
| `ll7` | `own_user` | Yes |
| Anyone else with a login | `untrusted` | No |
| Bot or GitHub Actions account | `untrusted` | No |
| Missing, null, deleted, or login-less author | `untrusted` (`missing_author`) | No |
| Foreign author plus an own-user flag | `flagged_by_own_user` | Yes, with attribution |

Classification is per row (issue/PR body, issue comment, PR comment, review
comment). The only exception is that an own-user issue/PR flag opts the whole
thread in, because the label cannot be attached to an individual comment.

## Explicit own-user flags

- Label `agent:digest` on the issue or pull request. The receipt records the
  flag source `label:agent:digest`; no author/timestamp is exposed by the label
  API, so those fields are `null`.
- An own-user comment or review body containing the marker
  `agent-digest: allow`. Like the label, the marker opts the whole thread in.
  The marker is honored only when the author is `ll7`; the receipt records the
  author login, timestamp, and row id. A marker inside foreign-authored text is
  ignored, so injection text cannot self-flag. Because the opt-in is
  thread-scoped, review the receipt attribution before trusting a thread where
  foreign comments were added after the marker.

## Receipt format

`receipt_for_rows` and the CLI emit a JSON-serializable
`agent_content_gate_receipt.v1` object:

```json
{
  "schema": "agent_content_gate_receipt.v1",
  "own_user": "ll7",
  "flags": [
    {"source": "comment_marker:agent-digest: allow", "author": "ll7",
     "created_at": "2026-09-02T12:00:00Z", "row_id": "123"}
  ],
  "included": [
    {"id": "body", "kind": "issue_body", "author": "ll7", "classification": "own_user",
     "reason": "own_user", "flag": null}
  ],
  "excluded": [
    {"id": "456", "kind": "issue_comment", "author": "mallory",
     "classification": "untrusted", "reason": "untrusted_author", "flag": null}
  ],
  "counts": {"total": 2, "included": 1, "excluded": 1}
}
```

Stable reasons: `own_user`, `untrusted_author`, `missing_author`,
`flagged_by_own_user`. The receipt never copies row bodies; it names rows,
authors, reasons, and flags so downstream context can stay compact and auditable.

## Fail-closed behavior

- Missing, null, login-less, or ambiguous author metadata is `untrusted`
  (`missing_author`) even when a flag is present; an unknown author cannot be
  attributed, so it cannot be included.
- Malformed fixtures and non-object rows make the CLI exit nonzero and never
  emit a trusted receipt.
- `read_complete_issue_thread` marks native `gh issue view` text as
  `auto_ingest_allowed: false` because native human output has no structured
  author metadata. The structured REST payloads carry per-row `author_trust`
  fields and a `trust_receipt`, and their generated digest excludes untrusted
  bodies.
- Issue batch capsules (`snapshot_issue_batch.py --capsule-dir`) blank the
  `body_excerpt` of untrusted rows and set `body_excerpt_excluded: true` with
  the reason and author attribution.

## CLI

```bash
uv run python scripts/dev/agent_content_gate.py --check --fixture rows.json --format json
uv run python scripts/dev/agent_content_gate.py --check --fixture rows.json --format markdown
```

The fixture is either a JSON row list or `{"labels": [...], "rows": [...]}`.
Exit code `0` means the receipt was built; `2` means the fixture was malformed.

## Wired surfaces

- `scripts/dev/gh_issue_rest.py`: `fetch_issue`, `fetch_issue_with_comments`
  add per-row `author_trust` plus `trust_receipt`; `view --plain --comments`
  and the REST thread fallback render trusted-only digests.
- `scripts/dev/snapshot_issue_batch.py`: explicit issue snapshots annotate
  `author_trust` and gate `body_excerpt`; `--capsule-dir` capsules blank
  untrusted bodies and add a `content_trust` block.

- `scripts/dev/snapshot_pr_queue.py`: review and comment snapshots annotate
  `author_trust` and blank excerpts from untrusted authors. The `agent:digest`
  PR label and own-user `agent-digest: allow` comment marker are thread-scoped
  opt-ins, so flagged foreign excerpts are retained with attribution.

- `scripts/dev/issue_audit_core.py`: issue body and comment source rows annotate
  `author_trust` and blank `text` for untrusted authors, so decision excerpts and
  documented options cannot quote foreign-authored instructions. The `agent:digest`
  label and own-user `agent-digest: allow` marker are thread-scoped opt-ins.

Not yet wired: worker prompt assembly outside these helpers. Treat its
foreign-authored text as unclassified until the gate is extended.
