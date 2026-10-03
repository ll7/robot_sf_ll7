# HYBDIAG Round 2 versus Round 3

AI-GENERATED/NEEDS-REVIEW. Development seeds only. Restoring the pedestrian
braking bound removes the earlier platform gain; zero collisions do not
prove unchanged pedestrian safety. Per-scenario intervals and all metrics
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
