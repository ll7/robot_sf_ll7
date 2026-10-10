# Goal
Prepare #10190 on origin/main; deliver a draft PR and bounded workstation smoke.
No default adoption, sealed/retired seeds, full study, Slurm dispatch, or merge.

# Contract and sources
Reuse the authored 0.0.8 matrix, 14-arm roster, horizons and native map metrics.
Profiles: legacy, radius, speed mean+spread, spread only, wall, combined.
Wall/combined depend on #10073; missing dependency must refuse execution.
Sources: #10190, #10074, #10073, #10000; literature citations in README.

# Steps and proof
1. Expose existing desired-speed settings and opt-in force radius in scenario overrides.
2. Prepare strict dev-only driver, paired bootstrap summaries and substrate probes.
3. Test runtime binding, seed exclusion and hand-calculated summary fixture.
4. Smoke one release scenario, goal/social_force, seeds 1001/1002, workers <=2.
5. Preserve per-episode rows, provenance, checksums, probes and summary; open draft PR.

# Claim and stop rules
Smoke proves wiring only; no ranking/realism/adoption claim. Fail closed on missing
profiles, malformed/unpaired rows or degraded planners. Full study starts only after
sealed main and doorway dispatch, and requires an explicit launch acknowledgement.
If #10073 is absent, test its real branch in an isolated integration checkout;
label dependency smoke separately and retain its exact source identity.

# Recovery
Fresh output directories refuse overwrite. Partial manifests remain marked running;
rerun in a fresh directory and retain interrupted artifacts. Local evidence remains
outside output/ and a compact public receipt is committed. No sub-agents.
