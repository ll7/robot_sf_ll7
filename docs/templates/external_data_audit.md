# External Data Audit Template

Use this template before staging external datasets, maps, checkpoints, generated fixtures, or other
third-party assets for Robot SF work. Keep paths repository-relative. Do not commit raw restricted
data unless the license and a maintainer decision explicitly allow redistribution.

## Summary

- Dataset or asset name:
- Upstream source URL:
- Access date:
- Upstream version, tag, commit, or release:
- Related issue or PR:
- Intended Robot SF use:

## License And Access

- Observed license:
- License URL or file:
- Access restrictions:
- Citation requirement:
- Citation text or BibTeX:
- Commercial, research-only, or non-redistribution limits:
- License compatibility decision: allowed / blocked / unclear
- Decision rationale:

## Download And Raw-File Policy

- Canonical download method:
- Required credentials or account:
- Expected raw files:
- Raw local cache path:
- Raw-file tracking decision: untracked / tracked fixture / external artifact pointer
- Raw-file redistribution decision: allowed / blocked / unclear
- Fail-closed behavior when raw files are missing:

## Derived Files And Fixtures

- Derived outputs to generate:
- Generation command or script:
- Derived output paths:
- Derived fixture tracking decision: untracked / tracked fixture / durable evidence copy
- Derived redistribution decision: allowed / blocked / unclear
- How derived files preserve license and citation metadata:
- Difference between derived fixtures and official source data:

## Checksums And Manifest

- Manifest path:
- Checksum algorithm: SHA-256
- Raw-file checksum requirement:
- Derived-file checksum requirement:
- Expected manifest fields:
  - source URL
  - access date
  - license
  - raw file names and checksums
  - derived file names and checksums
  - generation command
  - source commit
  - Robot SF use decision
- Verification command:

## Robot SF Use Decision

- Use status: allowed / blocked / reference-only / exploratory-only / benchmark-ready
- Benchmark eligibility: not eligible / blocked until hydrated / smoke-only / benchmark candidate
- Redistribution status: no redistribution / derived-only / raw-and-derived allowed
- Required durable pointer:
- Required follow-up issue:
- Reviewer notes:

## Validation Checklist

- [ ] Source URL, access date, and version are recorded.
- [ ] License, access restrictions, and citation requirements are recorded.
- [ ] Raw files, derived files, and redistribution decisions are separated.
- [ ] Checksums or an explicit checksum plan are recorded.
- [ ] The Robot SF use decision is one of allowed, blocked, reference-only, exploratory-only, or benchmark-ready.
- [ ] Missing artifacts fail closed with an actionable message.
- [ ] No restricted raw data is committed unless explicitly allowed.
