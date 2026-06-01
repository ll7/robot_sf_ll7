# Issue #2001 AMV Actuation Proxy Source Analysis

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/2001>
- <https://github.com/ll7/robot_sf_ll7/issues/1585>
- <https://github.com/ll7/robot_sf_ll7/issues/1559>

Related context:

- [`issue_1546_amv_actuation_envelope_stress_slice.md`](issue_1546_amv_actuation_envelope_stress_slice.md)
- [`issue_1556_amv_actuation_stress_slice.md`](issue_1556_amv_actuation_stress_slice.md)
- [`artifact_evidence_vocabulary.md`](artifact_evidence_vocabulary.md)

## Decision

Recommendation for #2001: `accepted_platform_class_proxy`.

Use the public TRL report, "In-Depth Investigation of E-Scooter Performance", as the preferred
platform-class proxy for **longitudinal acceleration and braking/deceleration only**. This is not a
hardware-calibrated AMV source and must not be used to claim AMV truth. It is acceptable as a weak
platform-class proxy when the report is cited directly, the missing fields are carried forward, and
any derived profile is named separately from the synthetic `amv-actuation-stress-v0` profile.

This decision does not close the calibrated-evidence gate in #1559. It gives #1585 a recoverable
proxy-source decision while preserving the stronger block on paper-facing calibrated AMV claims.

## Candidate Ranking

| Rank | Candidate | Source class | Access / license status | Supported fields | Missing fields | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | TRL, "In-Depth Investigation of E-Scooter Performance" | Public e-scooter performance report | Public PDF; redistribution terms not asserted here, so stage a local copy or manifest only through `amv-calibration` provenance. | Acceleration, braking/deceleration, peak speed context. | Yaw rate, angular acceleration, latency/update rate. | Preferred longitudinal proxy. |
| 2 | Li, Kovaceva, and Dozza, "Modeling collision avoidance maneuvers for micromobility vehicles" | Open-access micromobility field experiment | Open-access article; use citation and local manifest before staging source artifacts. | Braking/deceleration in comfortable/harsh maneuvers; steering-angle/lateral-offset context. | Standstill acceleration envelope, yaw rate, angular acceleration, latency/update rate. | Useful corroboration, not primary profile source. |
| 3 | Cano-Moreno et al., "E-scooter accelerations & speed database" | Public e-scooter acceleration/speed dataset | Zenodo dataset with DOI and per-file checksums visible on the record page; confirm license metadata during staging. | Real e-scooter acceleration records and longitudinal speed for a CityCross scooter. | Braking/deceleration summary, yaw rate, angular acceleration, latency/update rate. | Useful raw-data supplement, not primary source. |
| 4 | Garman et al., "Micro-Mobility Vehicle Dynamics and Rider Kinematics during Electric Scooter Riding" | SAE technical paper | Recoverable DOI page, but likely paywalled/manual-access source. | Instrumented e-scooter acceleration/velocity, steering angle, roll angle, GPS position. | Publicly recoverable numeric envelope values from the accessible page; latency/update rate. | `manual_license_review_required` if used beyond citation. |

## Preferred Source Details

TRL report:

- URL: <https://www.trl.co.uk/uploads/trl/documents/ACA104---In-Depth-Investigation-of-E-Scooter-Performance_1.pdf>
- Owner / publisher: TRL.
- Date / version: public PDF report, accessed 2026-06-01.
- Citation requirement: cite report title, owner, URL, and access date in downstream notes.
- Claim boundary: `platform_class_proxy`, not `hardware_calibrated_amv`.

Backed values from the TRL report:

| Field | Backed value | Notes |
| --- | --- | --- |
| Acceleration | Mean full-sample acceleration `2.819 m/s^2`; 500 W group mean `4.625 m/s^2`; lower-power groups around `2.425-2.664 m/s^2`. | Based on extracted acceleration phases from 113 acceleration runs. Use a conservative profile value only after choosing whether the proxy should represent broad e-scooter fleet behavior or heavier/higher-power devices. |
| Braking/deceleration | Mean full-sample deceleration `-3.429 m/s^2`; group means from about `-3.210` to `-3.613 m/s^2`; foot-brake sample `-3.842 m/s^2` but only `n=2`. | Based on 102 deceleration runs. Prefer the full-sample mean or grouped value with sample-size caveat; do not treat the sign convention as interchangeable without documenting it. |
| Yaw rate | Not supported. | Do not infer from steering, lateral offset, or scenario trajectories. |
| Angular acceleration | Not supported. | Do not infer from yaw-rate-free summaries. |
| Latency/update rate | Not supported. | Keep existing synthetic latency/update stress categories separate. |

## Corroborating Source Details

Li, Kovaceva, and Dozza:

- URL: <https://research.chalmers.se/publication/537916/file/537916_Fulltext.pdf>
- Article: "Modeling collision avoidance maneuvers for micromobility vehicles".
- Access / license note: open-access PDF; the PDF states an Elsevier / National Safety Council
  article under CC BY 4.0. Verify publisher metadata before redistributing a local copy.
- Useful fields: e-scooter braking/deceleration in comfortable and harsh maneuvers, plus steering
  angle and lateral-offset context for collision avoidance.
- Limitation: speed was reconstructed from LIDAR/trajectory post-processing after on-vehicle IMU
  synchronization loss, so this should corroborate braking behavior rather than replace the TRL
  source for a simple longitudinal proxy.

Cano-Moreno et al. Zenodo dataset:

- URL: <https://zenodo.org/record/6977206>
- DOI: <https://doi.org/10.5281/zenodo.6977206>
- Version: `1.0`, published 2022-08-09.
- Useful fields: vibration/acceleration records and longitudinal speed for a CityCross e-scooter in
  ECO and maximum-speed modes.
- Limitation: better suited to raw acceleration/speed and comfort analysis than to a concise
  actuation-envelope table; license metadata should be confirmed before staging.

Garman et al. SAE paper:

- DOI: <https://doi.org/10.4271/2020-01-0935>
- Useful fields from accessible abstract: a commercially available e-scooter was instrumented for
  acceleration, velocity, steering angle, roll angle, and GPS position.
- Limitation: accessible page does not expose the numeric envelope values needed for this issue;
  use requires manual access/license review.

## Claim Boundary For #1585 / #1559

Supported decision for #1585:

```yaml
source_decision: accepted_platform_class_proxy
claim_boundary: platform_class_proxy
preferred_source: TRL In-Depth Investigation of E-Scooter Performance
supported_actuation_fields:
  acceleration: true
  braking_deceleration: true
  yaw_rate: false
  angular_acceleration: false
  latency_update_rate: false
not_hardware_calibrated_amv: true
```

Recommended downstream interpretation:

1. A future proxy profile may use TRL-backed longitudinal values only if it carries
   `source_type: platform_class_proxy_source` and `claim_boundary: platform_class_proxy`.
2. Do not call the profile calibrated, paper-facing, hardware-aligned, or AMV-validated.
3. Keep yaw-rate, angular-acceleration, and latency/update-rate values as synthetic placeholders or
   `not_available` until a direct source is found.
4. If a local copy of the TRL PDF, Zenodo dataset files, or an extracted source manifest is staged,
   use the repository helper:

```bash
uv run python scripts/tools/manage_external_data.py stage amv-calibration \
  --source <local-file-or-dir>
```

The generated manifest belongs under ignored local output unless a compact, reviewable pointer is
explicitly promoted:

```bash
uv run python scripts/tools/manage_external_data.py check amv-calibration \
  --source <local-file-or-dir>
```

## Validation

- Verified the candidate source URLs are recoverable on 2026-06-01.
- Copied only values and units visible in the cited public source pages/PDF text.
- Verified the repository staging helper advertises `amv-calibration` and ties it to #1585/#1559:
  `uv run python scripts/tools/manage_external_data.py explain amv-calibration`.
