# Issue #10112 MINTORDER proof packet (2026-10-03)

Diagnostic-only source wiring and dev-seed rehearsal. Producer d56092ed remains explicit; round-1 fixes do not relabel old acquisition, name a freeze or grant scientific/production admission. Raw episodes and bundles are not in Git. Their hashes, including scored archive 437431ac4bd89b0fed0e15e1bbfd8942d1a672ce556e97a305210d3c5df3c04b and anchors dfb48e6846665bd8aa0ce6fdbf3043018ebeafb829347e6b23e47b42c37bab7b, bind external custody.

- [Rehearsal](rehearsal-d56092ed-proof.json): 2,016 values/terms/force records recompute, six declared ORCA branch rows, 98 archive files match.
- [Calibration](calibration-d56092ed-grid-proof.json): exact 14×48×2 dev grid and producer/config hashes.
- [Anchor attachment](actual-anchor-binding-d56092ed-preflight-cwd-fixed.json): raw-custody rederivation and artifact-only intake.
- [Dev1003 smoke](dev1003-d56092ed-proof.json): 14 admitted runtime-contract rows, zero excluded/degraded; no production admission.
- [Identity ledger](identity-differences.md), [test value](test-value.md), [failure classification](failure-classification.md), [red output](round1-red-witnesses.txt).

Regenerate numeric proofs with the committed script, supplying preserved external inputs (place all inputs/outputs inside your owned workspace). It reads rows and never resets/steps an environment:

```bash
uv run python scripts/dev/build_mintorder_evidence.py --output-dir "$PROOF_DIR" raw \
  --repository-root "$PRODUCER_CHECKOUT" --custody-root "$REHEARSAL_CUSTODY" \
  --smoke-root "$SMOKE_CUSTODY" --smoke-inventory "$SMOKE_INVENTORY"
```

Use the clean producer checkout at d56092ed for source/config pins. Custody contains calibration/, campaign/, acquired-anchors.json and the publication bundle/archive; smoke custody contains campaign/, dev1003-public-contract.json, checkpoint staging receipt and its inventory. The repository snapshot and external paths are operator inputs; no machine locator is committed. The generator checks every raw producer hash, grid cell, score, paired algorithm route and archive member; failed assertions refuse output. The public rehearsal projection drops only the private bundle locator; numeric values and producer SHA remain unchanged.

For identities use the script's `identity` subcommand with old/new resolver envelopes (c979e033/d56092ed, diagnostic DOI pair 99000001/99000002) and doorway templates/configs exported with `git show` from those same commits. It checks all 18 main differences and the four extra doorway field groups. For Markdown and red-log summaries, `project --input-dir "$REVIEW_NOTES" --names ...` writes portable reviewed inputs and checksum sidecars; it rejects private locators. Re-run sidecars and the registry baseline together after changing content. No CSV or raw bundles are included.

Existing repository tests may reset 7/42; this is allowed and not a blocker. New probes use only development seeds. The only remaining release gate is the documented post-merge freeze/acquisition/review/smoke/final-mint sequence.
