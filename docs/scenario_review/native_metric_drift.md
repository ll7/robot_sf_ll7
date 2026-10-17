# Native metric drift coverage

Future release records must distinguish same-head horizon preservation from an
absolute native-episode drift alarm. [Issue #10260](https://github.com/ll7/robot_sf_ll7/issues/10260)
tracks the unexplained platform difference and the remaining acceptance work.

The sealed 0.0.8 decision ledger names the deleted
`test_historical_runner_cap_matches_main_oracle`. Its current replacement is
`tests/benchmark/test_campaign_horizon_authority.py::test_historical_runner_cap_matches_paired_horizon_path`,
introduced by [PR #10227](https://github.com/ll7/robot_sf_ll7/pull/10227).
The sealed ledger remains unchanged. The replacement compares two horizon paths
at one source revision; shared planner, physics, dependency or scorer changes
can affect both sides and still pass. Synthetic metric sentinels protect a few
scorer definitions, and same-environment replay protects repeatability. Neither
establishes absolute native cross-platform stability.

For release 0.1.0 and successors, the decision-ledger tests resolve qualified
`tests/...py::test_name` references in **Enforced by** against current source.
They reject missing files and deleted functions without importing or running
the referenced tests. Historical ledgers receive their existing structural
checks against the preserved record; they are not reinterpreted against moving
main. Unqualified prose references and whether a live test proves a scientific
claim remain review obligations.

An absolute native drift alarm remains incomplete. Before choosing tolerances,
reproduce the development fixture at identical source, configuration, dependency
lock and native backend on both platforms, preserve metric and environment
provenance, and explain the variation. Then prove that the selected alarm accepts
measured variation and rejects a controlled out-of-tolerance change. Reference
updates require the same evidence and review; a moving-main literal or an
unexplained tolerance cannot supply this proof. This note admits no release
outcome or drift threshold.
