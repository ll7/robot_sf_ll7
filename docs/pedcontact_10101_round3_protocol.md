<!-- AI-GENERATED (#10101) - NEEDS-REVIEW -->
# PEDCONTACT Round 3 protocol corrections

AI-GENERATED / NEEDS-REVIEW

Reviewed head: 69753f16548add94a6ade93964d2a3f52e811ae9. The previous two-sided V6 ranking and whole-crowd fallback biased ranking are withdrawn.

A: Endpoint retry preserves velocities after convergence. Only unresolved pair/wall participants can stop; admissible rollback is local. Three jammed bodies plus a remote walker preserve the walker's (1.3,0) m/s velocity.

B/D: Keep .05rad/s sustained .3s. Huber section4.4 defines own-X distances about synchronized PoMD, a -3 to +3m analysis region, and human baseline-derived angular-speed onset; Fig4C marks the start of turning. The .05rad/s criterion is the author-ruled deterministic engineering equivalent, not a quoted universal human threshold. Per the 2026-10-03 ruling, published 2.1/2.4/2.7m are lower-bound comparisons. A turn already underway at the interpolated -3m boundary is right-censored >=3m; later turns retain measured onset distance. No upper acceptance limit. Source: [Huber 2014 section4.4, Fig4C, Table1](https://doi.org/10.1371/journal.pone.0089589).

The active opt-in config uses pedval.config.v2 / fixed_yaw_source_window_lower_bound_v3. Exact previous bytes remain readable in pedestrian_validation_0_0_9_history_v1.json. Only the research gate and source runner refer to the active config; the 0.0.8 campaign does not. This author-approved nonrelease change is enumerated separately from all released config/schema/golden byte comparisons.

C: Complete finite-N flow is N/((last-first)*width). Incomplete capture implies last > capture_end, so completion flow lies in [0,N/((capture_end-first)*width)]. Publish the upper bound, count, duration, censoring flag, observed-window throughput and prefix-rate diagnostic. A bound below the acceptance band proves FAIL; otherwise CENSORED, excluded from the producible denominator. Censoring cannot establish PASS; no first crossing leaves the bound unavailable. [Seyfried finite-N protocol](https://arxiv.org/abs/physics/0702004).

E: bounded_edge_v1 is the compatibility selector; the effective force is legacy_far_field_edge_correction_v2. The additive local correction is evaluated stably as bounded_edge + smoothstep(clearance,R,outer)*legacy, outer=max(1m,R+.5m). Exact legacy far field beyond outer; bounded nearest-surface response near bodies; zero endpoint derivatives. The manifest records effective version, selector, composition, outer clearance, baseline law/parameters, physical radius, and actual near-wall parameters. Robot arms off/contact/wall/both all use .28m.

F: The committed reviewed-head full-suite list includes every test ID and exception class. All254 pre-RNG/reset guard failures repeat on exact CALFIT base e6d52b556e2b0f82ccb0a6efd2c7ac17fffa6e36 with the same guard; matching command and receipts are committed. No protected seed was reset or stepped.

Test-value questions: jam detects unrelated velocity loss, missed by the old geometry-only cap test; analytic turn detects boundary censoring, missed by the old25m translation test; gate fixture detects the incorrect upper acceptance limit, previously asserted as ±20%; actual force detects lost far field, missed by near-edge tests; hand-counted crossing trace detects discarded partial completion, previously null until60/60; config checks effective declaration and preserved historical bytes. These use real paths/config bytes and hand-known answers independent of implementation. All seven witnesses fail at69753 for the intended reason.

Custody: older trajectories may use lossless temporal-XOR archives. Every original NPZ ZIP byte is reconstructed and SHA-256 verified before deletion; no precision is discarded. Indexes and decoder sources remain available.
