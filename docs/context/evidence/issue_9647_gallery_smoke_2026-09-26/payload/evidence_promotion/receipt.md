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
