# Release 0.0.8 decisions

## D-050 — Historical runner cap and authored 0.0.8 budgets (corrected 2026-09-30)

This ruling replaces the earlier premise that historical H600 extended authored
400/500-step scenarios. Measurements on main `93ba0d75`, the 2026-08 release
base `cd831d75` and radius campaign `aabad2e2` show that 600 only capped the
runner loop. The simulator stopped at its own authored limit; the effective
budget was `min(authored, 600)`, with authored-limit timeouts labelled
`terminated`. Historical configs through 0.0.7 must reproduce main exactly,
including every stable row field and historical episode identity.

Use `legacy_runner_cap`; the former policy name is removed without an alias.
The production exact-content SHA-256 registry maps each immutable historical
config to its true protocol version and this policy. Do not edit historical
config bytes or inject admission only in tests. The 0.0.8-cycle
`runtime_smoke_v0_4` is excluded. Provenance records the authored limit, runner
horizon 600, and effective minimum. The simulator limit is never raised by the
legacy policy. Input horizon provenance keys are reserved; a passed horizon
must match its binding.

For 0.0.8+, preserve the authored budgets (25×400, 13×500, 8×600, 1×650, 1×700).
A fixed horizon refuses shorter authored limits. Release acceptance must bind
the scenario schedule independently of the campaign and check every row's
budget; legacy-policy rows are forbidden. Reopen only with new measured
historical counterevidence or an explicit author ruling changing the protocol.

## D-054 — Measured v4 tuning budgets match 0.0.8 (corrected 2026-09-30)

The frozen v4 parameters come from #9748's 48-entry log (1,440 dev-seed episodes;
source `8eb0c386`, log SHA-256 prefix `1628d516`). The v1 tuning config declared
runner horizon 600, but the simulator used authored budgets: doorway medium
500, group crossing medium 500, perpendicular traffic 400, crowd navigation
400. Main's context resolution measures runner 600 and those simulator limits;
a native main crowd-navigation run on dev seed 1011 times out at exactly 400
with `terminated`. The budget code is unchanged from tuning source to main.

Those four effective tuning budgets equal 0.0.8's authored budgets. There is no
tuning-budget mismatch and no retuning is required on that premise. The v1 log's
86 timeouts (6 doorway, 14 perpendicular traffic, 66 crowd navigation, none group
crossing) reached 500/400/400, respectively. Keep the original inputs, log and
frozen parameters unchanged; v2 names the same budgets explicitly and any new
run has separate provenance. Authored-limit timeouts in 0.0.8 are `max_steps`
with horizon 400/500, whereas historical rows used `terminated` with runner
horizon 600. Station platform and double bottleneck get authored 650/700 in
0.0.8, versus the historical runner cap 600. Reopen only with new measured
counterevidence or a changed tuning/release protocol.
