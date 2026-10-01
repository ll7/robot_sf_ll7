<!-- AI-GENERATED (robot_sf#10064) - NEEDS-REVIEW -->
# Issue 10064: no-admissible recovery

AI-GENERATED NEEDS-REVIEW

Shared clearance-margin recovery fixes forced stasis when every executable command violates a hard constraint. Finite selection is retained. Recovery is best effort and does not certify admissibility or safety.

`validation.json` records fail-on-base witnesses, the full default suite and the one preexisting importer failure. `empty_world.json` preserves all 30 paired dev-only empty episodes; reset and command bytes match. `provenance.json` pins source, inputs, model bytes, scheduler limits and raw artifact digests. Raw validation logs and full trace archives are preserved at `~/admfix_evidence` and `imech192:~/admfix`.

The paired 48-scenario dev gate is pending in Slurm; no improvement or regression conclusion is available. Hosted CI is skipped while draft: ready would collect environment tests stepping forbidden111/123. No held-out evaluation or release claim.
