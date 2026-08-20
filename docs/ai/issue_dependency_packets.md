# Typed Issue Dependency Packets

`issue_dependency_packet.v1` is the canonical, machine-checkable record for prerequisites of one
issue. It turns a prose link such as “after pull request (PR) #42” into an exact predicate, a current
observation,
an admission verdict, and a precise unblock condition. It is workflow evidence only: it does not
establish scientific validity, artifact admissibility, licensing, compute authority, or release
readiness.

## One packet, not a second issue graph

An issue may embed the packet as a fenced JSON block or link one canonical JSON document. The
packet's `contract.source` and `contract.digest` bind it to the issue-contract text when that text
is available. Consumers should pass the exact packet to
[`scripts/dev/issue_dependency_packet.py`](../../scripts/dev/issue_dependency_packet.py); the helper
does not crawl arbitrary links, reconstruct dependencies from prose, or create a second graph of
issues.

The packet identifies the issue it serves and contains one row per predicate:

```json
{
  "schema": "issue_dependency_packet.v1",
  "repository": "ll7/robot_sf_ll7",
  "issue": 7613,
  "contract": {"source": "issue-body.md", "digest": "<sha256>"},
  "dependencies": [
    {
      "id": "implementation-pr",
      "repository": "ll7/robot_sf_ll7",
      "kind": "pull_request_state",
      "requirement": {"number": 42, "state": "MERGED", "head_sha": "<full-sha>"},
      "mandatory": true,
      "source": {"kind": "issue_contract", "ref": "acceptance-criteria-2"},
      "observed": {},
      "verdict": "unavailable",
      "unblock_condition": "PR #42 is merged at the named head SHA",
      "freshness": ["pull_request_state", "pull_request_head"]
    }
  ],
  "packet_digest": "<sha256>"
}
```

The eight supported dependency kinds are `issue_state`, `pull_request_state`, `commit_present`,
`path_present`, `artifact_digest`, `external_input`, `environment_capability`, and `human_ruling`.
Exact revisions, paths, schemas, digests, predicates, and ruling tokens are compared literally.
An issue or pull request being closed/merged never proves that an artifact exists.

## Evaluation semantics

The evaluator accepts explicit observations from a caller or a resolver. It emits one row for every
dependency and repeats the exact unblock condition for every failing row.

- `satisfied` means the named predicate and all requested exact fields match.
- `unsatisfied` means an observed predicate is false.
- `unavailable` means no named source verified the observation.
- `conflict` means an observation exists but an exact revision, digest, type, or token differs.
- `invalid` means the packet or predicate is malformed.

Mandatory failures produce `verdict: "blocked"` and `ok: false`. Advisory failures remain in
`advisory_failures` and do not block. Evaluation has no timestamp, so unchanged packet and context
inputs produce byte-stable JSON.

The resolver uses bounded read-only `gh api` calls for public issue, pull-request, commit, path,
and ruling reads. A supplied repository root enables read-only local path and Git ancestry checks.
Artifacts, external inputs, and environment capabilities remain unavailable unless a named,
already-verified observation is supplied. It never downloads, merges, closes, submits, labels, or
answers a human ruling.

## Consumer adapter

The #7611 implementability owner can attach the aggregate result without duplicating dependency
logic:

```python
from scripts.dev.issue_dependency_packet import apply_dependency_gate, resolve_packet

evaluation = resolve_packet(packet, repo_root=repo_root)
report = apply_dependency_gate(implementability_report, evaluation)
```

When the aggregate is not satisfied, the adapter sets both `ready` and `write_allowed` to `false`,
marks the classification as `needs_dependency` when appropriate, and preserves every failing row.
The adapter performs no mutation itself.

## Local validation

```bash
uv run python scripts/dev/issue_dependency_packet.py validate --packet packet.json
uv run python scripts/dev/issue_dependency_packet.py verify \
  --packet packet.json --context observations.json
uv run pytest -q tests/dev -k "issue_dependency or unblock_predicate"
```

This contract is generic workflow dependency integrity. It is not a replacement for campaign
answerability, evidence/result packets, benchmark provenance, or maintainer/domain approval.
