# Pedestrian gate round 2

Diagnostic-only. Exact cumulative head `d90bf2436e24dea33198a8bbaf6bbf4342b0edac`. Original dev1001 SinglePedestrianBehavior bind+step witness, 10 s start delay, dt 0.1 s, radius 0.28 m, initially separated actors; no robot instantiated.

Held displacement with `projection_v1`: **0.0 m**; velocity `[0.0, 0.0]`, cap 0, remaining delay 9.9 s. Contact-off control also 0.0 m. Previous head `52634e85` moved the held actor 0.05293751034406103 m. This closes this specific refutation of #10178; it does not establish a passing #10000 gate or robot hard-stop/yield correctness. PRs remain draft and blocked.

[Witness source](hold_probe.py), [output and provenance](hold_probe.json). Installed pysocialforce source was compared byte-for-byte to all 17 Python files at the pinned head. Original witness source bytes retained. Executed via `uv run --no-sync python <hold_probe.py>` in the exact-head review environment.

Targeted #10180 attribution is pending; no full sweep is scheduled.
