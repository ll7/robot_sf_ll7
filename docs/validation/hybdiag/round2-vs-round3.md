# HYBDIAG Round 2 versus Round 3

AI-GENERATED/NEEDS-REVIEW. Development seeds only. The platform experiment
is historical and has been removed; zero collisions do not prove unchanged
pedestrian safety. Per-scenario intervals and all metrics
are in the corresponding `round*-results.csv` files.

| Round/world | Arm | S/C/T | Success Wilson 95% | Collision Wilson 95% | Timeout Wilson 95% | Freeze | Stopped % | No moving s | Min ped m | Near events; per 1,000 robot-s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2/crowd | off | 235/0/65 | 73.3–82.6% | 0.0–1.3% | 17.4–26.7% | 35/300 | 19.44 | 1739.2 | 1.593 | 255; 27.34 |
| 2/crowd | static | 270/0/30 | 86.1–92.9% | 0.0–1.3% | 7.1–13.9% | 0/300 | 1.47 | 21.2 | 1.593 | 276; 34.03 |
| 2/crowd | platform | 261/0/39 | 82.7–90.3% | 0.0–1.3% | 9.7–17.3% | 33/300 | 21.70 | 1682.8 | 1.589 | 526; 65.04 |
| 2/crowd | both | 291/0/9 | 94.4–98.4% | 0.0–1.3% | 1.6–5.6% | 0/300 | 1.66 | 15.1 | 1.530 | 553; 82.19 |
| 2/empty | off | 89/0/13 | 79.4–92.4% | 0.0–3.6% | 7.6–20.6% | 4/102 | 10.31 | 202.0 | — | 0; 0.00 |
| 2/empty | static | 100/0/2 | 93.1–99.5% | 0.0–3.6% | 0.5–6.9% | 2/102 | 7.41 | 98.4 | — | 0; 0.00 |
| 2/empty | platform | 89/0/13 | 79.4–92.4% | 0.0–3.6% | 7.6–20.6% | 4/102 | 10.31 | 202.0 | — | 0; 0.00 |
| 2/empty | both | 100/0/2 | 93.1–99.5% | 0.0–3.6% | 0.5–6.9% | 2/102 | 7.41 | 98.4 | — | 0; 0.00 |
| 3/crowd | off | 235/0/65 | 73.3–82.6% | 0.0–1.3% | 17.4–26.7% | 35/300 | 19.44 | 1739.2 | 1.593 | 255; 27.34 |
| 3/crowd | static | 270/0/30 | 86.1–92.9% | 0.0–1.3% | 7.1–13.9% | 0/300 | 1.47 | 21.2 | 1.593 | 276; 34.03 |
| 3/crowd | platform | 236/0/64 | 73.7–82.9% | 0.0–1.3% | 17.1–26.3% | 34/300 | 19.60 | 1711.8 | 1.530 | 330; 36.24 |
| 3/crowd | both | 270/0/30 | 86.1–92.9% | 0.0–1.3% | 7.1–13.9% | 0/300 | 1.31 | 10.5 | 1.593 | 370; 46.72 |
| 3/empty | off | 89/0/13 | 79.4–92.4% | 0.0–3.6% | 7.6–20.6% | 4/102 | 10.31 | 202.0 | — | 0; 0.00 |
| 3/empty | static | 100/0/2 | 93.1–99.5% | 0.0–3.6% | 0.5–6.9% | 2/102 | 7.44 | 98.8 | — | 0; 0.00 |
| 3/empty | platform | 89/0/13 | 79.4–92.4% | 0.0–3.6% | 7.6–20.6% | 4/102 | 10.31 | 202.0 | — | 0; 0.00 |
| 3/empty | both | 100/0/2 | 93.1–99.5% | 0.0–3.6% | 0.5–6.9% | 2/102 | 7.44 | 98.8 | — | 0; 0.00 |

Each crowded arm has 300 episodes; each empty arm has 102.

Per-scenario near-miss decomposition (crowded worlds). Events count entries
into the global-any-pedestrian surface-gap [0, 0.50) m predicate. Robot-seconds
are actual exposure. Different route completion changes encounter exposure.

| Round | Scenario | Arm | Events | Robot-seconds | Events / 1,000 robot-s |
| --- | --- | --- | --- | --- | --- |
| 2 | ALL | off | 255 | 9326.4 | 27.34 |
| 2 | ALL | static | 276 | 8110.8 | 34.03 |
| 2 | ALL | platform | 526 | 8087.1 | 65.04 |
| 2 | ALL | both | 553 | 6728.2 | 82.19 |
| 2 | classic_station_platform_medium | off | 102 | 1800.0 | 56.67 |
| 2 | classic_station_platform_medium | static | 87 | 1800.0 | 48.33 |
| 2 | classic_station_platform_medium | platform | 237 | 1676.7 | 141.35 |
| 2 | classic_station_platform_medium | both | 227 | 1707.1 | 132.97 |
| 2 | classic_cross_trap_high | off | 51 | 1221.0 | 41.77 |
| 2 | classic_cross_trap_high | static | 50 | 1259.9 | 39.69 |
| 2 | classic_cross_trap_high | platform | 104 | 865.7 | 120.13 |
| 2 | classic_cross_trap_high | both | 101 | 844.5 | 119.60 |
| 2 | classic_t_intersection_medium | off | 8 | 629.6 | 12.71 |
| 2 | classic_t_intersection_medium | static | 12 | 641.5 | 18.71 |
| 2 | classic_t_intersection_medium | platform | 15 | 533.1 | 28.14 |
| 2 | classic_t_intersection_medium | both | 21 | 510.1 | 41.17 |
| 2 | francis2023_narrow_doorway_width_2p20 | off | 0 | 1800.0 | 0.00 |
| 2 | francis2023_narrow_doorway_width_2p20 | static | 30 | 521.1 | 57.57 |
| 2 | francis2023_narrow_doorway_width_2p20 | platform | 0 | 1800.0 | 0.00 |
| 2 | francis2023_narrow_doorway_width_2p20 | both | 30 | 459.7 | 65.26 |
| 2 | classic_cross_trap_low | off | 8 | 850.1 | 9.41 |
| 2 | classic_cross_trap_low | static | 12 | 888.1 | 13.51 |
| 2 | classic_cross_trap_low | platform | 30 | 732.3 | 40.97 |
| 2 | classic_cross_trap_low | both | 31 | 740.4 | 41.87 |
| 2 | classic_cross_trap_medium | off | 35 | 1122.9 | 31.17 |
| 2 | classic_cross_trap_medium | static | 34 | 1121.0 | 30.33 |
| 2 | classic_cross_trap_medium | platform | 67 | 810.0 | 82.72 |
| 2 | classic_cross_trap_medium | both | 66 | 823.7 | 80.13 |
| 2 | classic_t_intersection_low | off | 7 | 624.8 | 11.20 |
| 2 | classic_t_intersection_low | static | 8 | 607.6 | 13.17 |
| 2 | classic_t_intersection_low | platform | 9 | 530.0 | 16.98 |
| 2 | classic_t_intersection_low | both | 13 | 505.0 | 25.74 |
| 2 | classic_doorway_low | off | 24 | 362.4 | 66.23 |
| 2 | classic_doorway_low | static | 22 | 358.4 | 61.38 |
| 2 | classic_doorway_low | platform | 23 | 325.4 | 70.68 |
| 2 | classic_doorway_low | both | 23 | 319.6 | 71.96 |
| 2 | classic_group_crossing_low | off | 5 | 377.5 | 13.25 |
| 2 | classic_group_crossing_low | static | 4 | 375.1 | 10.66 |
| 2 | classic_group_crossing_low | platform | 17 | 330.4 | 51.45 |
| 2 | classic_group_crossing_low | both | 17 | 333.6 | 50.96 |
| 2 | classic_head_on_corridor_low | off | 15 | 538.1 | 27.88 |
| 2 | classic_head_on_corridor_low | static | 17 | 538.1 | 31.59 |
| 2 | classic_head_on_corridor_low | platform | 24 | 483.5 | 49.64 |
| 2 | classic_head_on_corridor_low | both | 24 | 484.5 | 49.54 |
| 3 | ALL | off | 255 | 9326.4 | 27.34 |
| 3 | ALL | static | 276 | 8110.8 | 34.03 |
| 3 | ALL | platform | 330 | 9105.9 | 36.24 |
| 3 | ALL | both | 370 | 7919.1 | 46.72 |
| 3 | classic_station_platform_medium | off | 102 | 1800.0 | 56.67 |
| 3 | classic_station_platform_medium | static | 87 | 1800.0 | 48.33 |
| 3 | classic_station_platform_medium | platform | 132 | 1800.0 | 73.33 |
| 3 | classic_station_platform_medium | both | 135 | 1800.0 | 75.00 |
| 3 | classic_cross_trap_high | off | 51 | 1221.0 | 41.77 |
| 3 | classic_cross_trap_high | static | 50 | 1259.9 | 39.69 |
| 3 | classic_cross_trap_high | platform | 71 | 1169.7 | 60.70 |
| 3 | classic_cross_trap_high | both | 67 | 1188.5 | 56.37 |
| 3 | classic_t_intersection_medium | off | 8 | 629.6 | 12.71 |
| 3 | classic_t_intersection_medium | static | 12 | 641.5 | 18.71 |
| 3 | classic_t_intersection_medium | platform | 12 | 626.4 | 19.16 |
| 3 | classic_t_intersection_medium | both | 14 | 616.2 | 22.72 |
| 3 | francis2023_narrow_doorway_width_2p20 | off | 0 | 1800.0 | 0.00 |
| 3 | francis2023_narrow_doorway_width_2p20 | static | 30 | 521.1 | 57.57 |
| 3 | francis2023_narrow_doorway_width_2p20 | platform | 0 | 1800.0 | 0.00 |
| 3 | francis2023_narrow_doorway_width_2p20 | both | 30 | 517.7 | 57.95 |
| 3 | classic_cross_trap_low | off | 8 | 850.1 | 9.41 |
| 3 | classic_cross_trap_low | static | 12 | 888.1 | 13.51 |
| 3 | classic_cross_trap_low | platform | 16 | 832.4 | 19.22 |
| 3 | classic_cross_trap_low | both | 17 | 865.9 | 19.63 |
| 3 | classic_cross_trap_medium | off | 35 | 1122.9 | 31.17 |
| 3 | classic_cross_trap_medium | static | 34 | 1121.0 | 30.33 |
| 3 | classic_cross_trap_medium | platform | 50 | 1023.6 | 48.85 |
| 3 | classic_cross_trap_medium | both | 50 | 1088.8 | 45.92 |
| 3 | classic_t_intersection_low | off | 7 | 624.8 | 11.20 |
| 3 | classic_t_intersection_low | static | 8 | 607.6 | 13.17 |
| 3 | classic_t_intersection_low | platform | 5 | 589.5 | 8.48 |
| 3 | classic_t_intersection_low | both | 10 | 575.5 | 17.38 |
| 3 | classic_doorway_low | off | 24 | 362.4 | 66.23 |
| 3 | classic_doorway_low | static | 22 | 358.4 | 61.38 |
| 3 | classic_doorway_low | platform | 22 | 348.4 | 63.15 |
| 3 | classic_doorway_low | both | 24 | 355.3 | 67.55 |
| 3 | classic_group_crossing_low | off | 5 | 377.5 | 13.25 |
| 3 | classic_group_crossing_low | static | 4 | 375.1 | 10.66 |
| 3 | classic_group_crossing_low | platform | 6 | 377.6 | 15.89 |
| 3 | classic_group_crossing_low | both | 5 | 376.9 | 13.27 |
| 3 | classic_head_on_corridor_low | off | 15 | 538.1 | 27.88 |
| 3 | classic_head_on_corridor_low | static | 17 | 538.1 | 31.59 |
| 3 | classic_head_on_corridor_low | platform | 16 | 538.3 | 29.72 |
| 3 | classic_head_on_corridor_low | both | 18 | 534.3 | 33.69 |
