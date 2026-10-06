# Real camera-ready dev fixtures

Diagnostic-only rehearsal data, not release or paper evidence. Source commit:
`ea414933e61ce267389bd3bcbe97fb669a825c6e` (rehearsal-only anchor and fallback
workarounds documented in the lane report). Every copied episode has seed **1001**.
No planner or environment execution on either evaluation band occurred.

The original JSON and their missing source companions live in
`rehearsal_snapshot.zip`, an archived diagnostic source namespace. Internal paths
refer to the rehearsal commit, not this checkout. The archive retains the exact
row/config bytes and `provenance.json` binds every member by SHA-256. This packages
source data without creating release policy at canonical config paths.

- Archive members `camera_ready_row.json` / `goal_row.json`: unedited first goal row from
  `reh_main_v2/runs/goal__differential_drive/episodes.jsonl`.
- Archive member `guarded_ppo_row.json`: unedited first guarded PPO row from that campaign.
- Archive member `pinned_runtime_rows.json`: configuration projection resolved from the pinned
  source/config for those slots, without model loading or simulator steps. Cached
  literals exercise the validator independently of a test-built identity envelope.
- Archive source companions: the original authored horizon schedule and the
  dev1001–1003 rehearsal campaign config, under `source/`. Neither is executed or
  adopted as current release policy.
- `rehearsal_traces.jsonl.gz`: unchanged JSONL lines from `reh_probe_traces_v2`,
  compressed with deterministic gzip metadata. Includes the ten flagged episodes
  and one clean goal episode (11 of 672). `provenance.json` records original line
  digests, run names, episode identities and fixture file hashes.

Real provenance uses `config_hash` and `git_hash`. Camera-ready scenario identities
omit `seed`; the seed is at row root and in seed-derived route defaults. Guarded
PPO records `algorithm: ppo`, `canonical_algorithm: guarded_ppo` and the typed PPO
configuration, while `scenario_params.algo_config_hash` binds the complete guard
configuration. Historical traces have no reset angular velocity; their first yaw
acceleration remains explicitly unavailable. Future traces retain that measured
state under `reset.robot.angular_velocity` (rad/s).

The checker supports exactly `simulation-step-trace.v1` and
`simulation-step-trace.v2`, one version per invocation. It rejects mixed versions
and marks unknown schemas unavailable without interpreting their trace geometry.

The release comparator remains a slot/seed-paired audit. It does not implement
independent-sample distribution comparison. The full rehearsal passes row
admission and stops at the strict release probe cardinality gate (42 dev-seed
slots versus 420 release slots). No gate or seed-pair policy was bypassed.
