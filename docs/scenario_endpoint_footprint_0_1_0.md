# Full-footprint endpoint dispositions for 0.1.0

Diagnostic-only. `inspect_release_zones()` and `--release-zones` default to
`robot_pedestrian_radii_v2`: centre-support distance must exceed the sum of
`config.robot_config.radius` and `config.sim_config.ped_radius` to be clear.
The resolved release values are 1.0 m and 0.4 m. The comparison includes
round endcaps and tangency with a 1e-9 m numerical allowance. No environment
or held-out seed is executed. These are possible footprint contacts over
full authored support, not observations that every sampled pose collides.

Fresh main has 53 historical pedestrian-radius findings. The new audit has
65: twelve additional findings in seven scenarios. The exact identities,
distances, polygons, radii and fingerprints are in
[the evidence JSON](scenario_hygiene/endpoint_footprint_0_1_0.json).

Each additional finding remains visible. The dispositions below retain the
geometry as a diagnostic input; they grant no new exact waiver, clearance
certificate, nominal benchmark eligibility or release approval. Endpoint
certification must address them with the effective contact/radius contract
before 0.1.0 benchmark promotion. Old release artifacts remain historical.

| Scenario | Additional contact support | Distance | Written disposition |
| --- | --- | --- | --- |
| `classic_station_platform_medium` | goal, single p1 | 1.0 m | Retain as station interaction diagnostic. p1's complete lane support approaches the full destination rectangle; center separation alone does not prove footprint separation. Require destination/lane separation or an explicitly approved interaction contract before nominal use. |
| `classic_head_on_corridor_low` | goal, ped spawn 0 | 1.274 m | Retain as active head-on spawn-overlap diagnostic (resolved density 0.02). The crowd spawn support can contact the destination footprint. Require goal/spawn separation or an explicitly approved interaction contract before nominal use. |
| `classic_head_on_corridor_medium` | goal, ped spawn 0 | 1.274 m | Retain as active head-on spawn-overlap diagnostic (resolved density 0.05). The same clearance gap applies as in the low variant; require geometry repair or an approved interaction contract before nominal use. |
| `francis2023_frontal_approach` | goal, single h1 | 1.0 m | Retain as opposite-direction interaction diagnostic. The lane can occupy the robot endpoint footprint; the authored full goal support is not certified safe by center-only separation. Require an approved endpoint interaction or geometry repair before nominal use. |
| `francis2023_pedestrian_obstruction` | goal, single h1 | 1.0 m | Retain as obstruction diagnostic. An occupied destination is a task risk; do not silently treat this as a clear-goal benchmark. Separate the endpoint or certify the intentional occupied-goal contract before nominal use. |
| `francis2023_robot_overtaking` | goal, single h1 | 1.0 m | Retain as overtaking diagnostic. Full lane support and goal footprint can contact even though centers clear. Require endpoint/lane separation or an approved overtaking interaction contract before nominal use. |
| `francis2023_parallel_traffic` | spawn: two crowd routes and two ped spawns; goal: two crowd routes | routes 0.5 m; spawn rectangles 1.0 m | Retain as parallel-traffic diagnostic with six explicit findings. These are active route populations, so no dormant-population disposition applies. Separation must include both radii and route-spawn jitter; adjust lane/endpoint supports with the 0.1.0 radius calibration before nominal use. |

The audit fingerprints contain both radii and the policy. Old 0.0.8
fingerprints therefore cannot waive these findings. The historical policy
`pedestrian_radius_v1` reproduces the old output and CI's old exact-waiver
contract explicitly, without changing any release disposition bytes.

The regression checks distance-only overlap at 0.5 m, tangency at 1.4 m,
and non-contact at 1.5 m through the real loader/auditor. The two positive
cases fail on base. Existing radius tests only prove pedestrian expansion.
The test uses effective runtime radii, adds no production seam, and is cheaper
than constructing environments or sampling collisions.

No simulator behavior or radius value changes here. Re-run this deterministic
audit after contact/radius calibration: changed effective radii invalidate its
fingerprints. #10175's authoring files are untouched; future successor inputs
need their own audit rather than inheriting release dispositions.
