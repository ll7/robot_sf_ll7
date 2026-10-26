# Goal

Build the Stage-3 0.0.7→0.0.8 comparison gate required by #9668. Every
available episode pair and every common outcome/metric is compared; changed
values may pass only with a named versioned input/source change and a checked
causal receipt. No real 0.0.8 row is available for admission yet.

# Scope

- New standalone comparator and focused synthetic tests in distinct paths.
- Bind the accepted 0.0.7 publication bundle SHA-256 `684da7c5…52a6fb2f`,
  source `07f7e8d…`, effective config `0953313…`, and executed v1 matrix
  SHA-256 `03fc8330…bb5c2c`. The stale tracked release manifest is not used.
- Require a 14-slot mapping to 0.0.8 keys; v4 replacements receive the literal
  `implementation replaced` label and new names containing `v4`.
- Keep historical files, tag, bundle, and Zenodo record unchanged. This packet
  does not submit evaluation jobs, tag, publish, or admit Chapter 7 claims.

# Evidence sources

- #9668 author ruling and Stage-3 contract; accepted 0.0.7 closeout
  `docs/analysis/issue_9431_release_0_0_7_closeout.md`.
- `scripts/analysis/compare_issue_9431_release.py` historical roster and
  archive layout. The old unmerged #9668 equivalence tool is reference only;
  its strict equality policy is superseded.
- Public Zenodo 0.0.7 carrier cold readback at
  `docs/analysis/issue_9668_stage3_007_cold_readback.json`: exact archive,
  source, effective config, and executed v1 matrix pins; 20,160 rows,
  14 arms, 48 scenarios, 30 seeds, zero reader anomalies. Legacy non-finite
  sentinels and 60 intentional hybrid runtime ORCA selections are recorded.

# Steps

1. Parse the pinned archive and a checksummed candidate root without extracting
   the archive. Require source/config/matrix identities and exact arm-slot map.
   Resolve every scenario's effective algorithm from the hashed campaign and
   hybrid configs, including the only approved ORCA hand-off.
2. Inventory missing, extra, malformed, and duplicate identities while pairing
   every available valid historical/candidate row. Audit raw candidate execution
   markers and integrity before row compaction; do not apply new admission
   policy retroactively to accepted 0.0.7 rows.
3. Compare all common outcome and metric fields at absolute `1e-12`; treat every
   change as a finding. Verify one explicit attribution ledger entry per finding
   against a checksummed causal receipt and a versioned change declared by the
   candidate. Missing attribution blocks; explained corrections need not equal
   0.0.7. Quantify paired rates and rank shifts.
4. Run synthetic fixtures and focused tests, then deliver a draft PR for
   independent exact-head review. No campaign rows are manufactured as evidence.

# Decisions and risks

- A checked receipt proves binding and a recorded causal review decision, not
  the truth of the scientific mechanism. Domain review of receipt contents and
  source/config hashes remains a release gate.
- The observed collision rate uses `outcome.collision_event`, not a planner-caused
  contact rate. That interpretation needs the separate #9729 attribution gate.
- Partial or duplicate matrices always block admission, but the diagnostic
  report still contains all available pairs and their changed fields.
- The 18-row H400 doorway slice has a separate manifest and is excluded here.

# Validation route

Focused comparator tests, Ruff check/format, diff check, exact-diff self-review,
and independent final-head review. No broad readiness or archive extraction on
this disk-constrained host.

# Recovery / handoff

Isolated branch `issue-9668-stage3-comparison-20260928` starts at
`origin/main` `4665cd13`; retain report fixtures only in tests and keep draft
until a real candidate and causal receipts exist.
