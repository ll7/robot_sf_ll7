# #9647 one-case historical replay and rendering smoke

- **Executed:** 2026-09-26
- **Runner code revision:** `4b9617338ecbaf6bd3612b4d8c5269f993581a0a` (based on `origin/main` `7815da2daaf646d8fb6d3bada1ab8c31fb2d3572`)
- **Environment:** Python 3.13.14; one worker; replay batch runtime 5.7286 seconds.
- **Source fixture:** tracked #1501 `failure_0002` compatibility fixture. This was not produced by the #9645 search and is not a new discovery.
- **Fixture search metadata:** Optuna, seed 42, budget 32; the compatibility manifest represents only one candidate row.
- **Command:** `scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/materialize_adversarial_replay_gallery.py tests/fixtures/adversarial_replay_gallery/issue_1501_compat/manifest.json --out output/adversarial-replay-gallery/issue-9647-smoke-4b96173 --top-k 1`
- **Result:** one case selected and replayed. Scenario `crossing_ttc_template_adversarial_0008`, seed 320, planner `goal`; identity, categorical outcome, failure attribution, and objective matched. Both source and replay report a collision (`collision_count=1`, route incomplete; objective `4.833333333333333`). The verification status is `outcome_reproduced_revision_changed`: replay code `4b961...` differs from the source episode revision `58e516...`.
- **Feasibility:** a post-hoc `scenario_cert.v1` static-route certificate passed as `hard_but_solvable`. It does not establish dynamic task feasibility, which remains unknown. The source episode map-registry digest is absent, so full source input binding is unknown.
- **Figures:** [critical frame](../cases/case_0000_01595b2ea612/figures/still_9.png), [one-frame filmstrip](../cases/case_0000_01595b2ea612/figures/filmstrip.png), and [trajectory](../cases/case_0000_01595b2ea612/figures/trajectory.png). The existing renderer did not load the SVG map overlay, and the canonical runner emitted no video artifact; both are recorded as unavailable.
- **Claim boundary:** this demonstrates selector, replay-comparison, and existing-renderer plumbing on one historical fixture. It does not demonstrate a newly discovered counterexample, exact-source reproduction, a planner comparison, or simulator/real-world safety.


## Render-only visual follow-up

- **Renderer revision:** `d0a8ccf822c48ae7d61dae361440e7335ee43013`. This followed the original one-case replay and consumed only the promoted `simulation-step-trace.v1`; it did not run a simulator, replay dynamics, search, or campaign.
- **Trace custody:** [stored trace](../cases/case_0000_01595b2ea612/replay/simulation_step_trace.json), SHA-256 `d12df9eb2030a29eae8cea8cc1bc5b042c9865bd9b9afcca8d83ed3da5872788`; [trace provenance](../cases/case_0000_01595b2ea612/replay/trace_provenance.json), SHA-256 `5f2ac197e51e629c023c1ecbbb669d4f2ccc69e6c5bc4ee2b211c21e43407d23`. The source episode row SHA-256 is `cd5597cec3728b27379c21e10434c9ed17aaa0f65937a98f24fb4bd9d692cdc8` and its original run-sidecar SHA-256 is `5d54d9167b49c57e04f7a04ce87e220edac5c32a85eab5cf7db4369b9f28a2e0`. The full JSONL row and ignored output tree were not copied.
- **Render command:** `uv run python scripts/tools/render_recorded_replay_trace.py --trace docs/context/evidence/issue_9647_gallery_smoke_2026-09-26/payload/cases/case_0000_01595b2ea612/replay/simulation_step_trace.json --provenance docs/context/evidence/issue_9647_gallery_smoke_2026-09-26/payload/cases/case_0000_01595b2ea612/replay/trace_provenance.json --out output/adversarial-replay-gallery/issue-9647-render-only-20260926`
- **Frames:** the deterministic filmstrip samples steps `[0, 2, 4, 7, 9]`; the critical still is step 9. A second render into a different output directory produced byte-identical PNGs.
- **Separate diagnostics:** trace sample `simulation_step_trace.steps[9].pedestrians[0].surface_clearance_m` is `-0.015686402836 m` at t=1.0s. The exact event ledger reports collision at t=1.0s from `runtime.step.meta.is_pedestrian_collision`. Its nearest recorded sample is step 9 at t=1.0s. These two sourced facts coincide at step 9 in this case; the clearance value is not used to infer the collision event.
- **Video:** unavailable in this environment: pygame is missing and MoviePy is unavailable. The existing video renderer also has no collision/clearance overlay support. The static filmstrip and trajectory include separate collision and minimum-clearance symbols and labels.
- **Map:** unavailable for the existing static image reader; no source map asset was passed to this render-only command.
- **Interpretation:** these figures are visualizations of stored positions. They do not run or verify replay determinism, change the original `outcome_reproduced_revision_changed` result, establish dynamic feasibility, resolve unknown source map-registry binding, or support safety claims.

See [`render_only_provenance.json`](render_only_provenance.json) for machine-readable sources, frame selection, output hashes, repeat-render comparison, and limitations.
