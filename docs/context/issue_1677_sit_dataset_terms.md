# Issue #1677 SiT Dataset Terms Audit

Date: 2026-05-29

Parent issue: <https://github.com/ll7/robot_sf_ll7/issues/1677>

Parent survey: <https://github.com/ll7/robot_sf_ll7/issues/1617>

Evidence manifest:
[`evidence/issue_1677_sit_dataset_terms.json`](evidence/issue_1677_sit_dataset_terms.json)

## Decision

Treat SiT Dataset as an external pointer only. Robot SF should not commit raw, transformed,
sampled, or derived SiT dataset fixtures until the effective dataset license and redistribution
boundary are clarified by maintainers or by explicit project legal review.

This is a conservative data-governance decision, not a judgment that SiT is unusable for research.
It remains useful as a source reference for social-navigation scenarios, robot-centric pedestrian
trajectory prediction, semantic-map ideas, and benchmark structure.

## Source Observations

* Repository: <https://github.com/SPALaboratory/SiT-Dataset>
* Observed upstream commit: `b5fa5ae795dc0041468d07b997b0d50c166fb76e`
  (`2024-10-17T09:16:53Z`, README update).
* Project page: <https://spalaboratory.github.io/SiT/>
* Paper page: <https://openreview.net/forum?id=gMYsxTin4x>
* Download form:
  <https://docs.google.com/forms/u/3/d/e/1FAIpQLSfsMQq5Ob2NI1aV96C7NhnMtXvBshJvbuegVeQ5B3KETNv0FQ/viewform?usp=send_form>

The repository README says the final dataset was released in June 2024, the Google Form link was
changed in October 2024, semantic map data is available in the GitHub repository, and the full
dataset download is reached through the Google Form.

The project page describes the dataset as robot-centric social-navigation data collected with a
Clearpath Husky UGV, two Velodyne VLP-16 LiDARs, five Basler cameras, IMU/GPS sensors, and
annotations for 2D and 3D boxes, tracking, trajectory prediction, and end-to-end motion forecasting.

The OpenReview paper page records the NeurIPS 2023 Datasets and Benchmarks paper title, authors,
publication date, abstract, and project URL.

## License And Terms

Observed code license:

* The GitHub repository metadata and `LICENSE` file are Apache-2.0.

Observed dataset terms:

* The README states that the SiT dataset is under Creative Commons BY-NC-ND 4.0 and code is under
  Apache-2.0.
* The current Google Form terms say dataset access is non-commercial and, unless otherwise stated,
  the datasets are under Creative Commons BY-NC-SA 4.0 plus additional terms. The form states that
  its dataset terms take precedence if there is a conflict.
* The form asks users to agree to the terms and provide name, affiliation, country, and email before
  access. It also reserves the provider's right to modify terms.

Conservative interpretation:

* Dataset assets are non-commercial.
* The public sources disagree about whether derivatives are disallowed (`ND`) or allowed only under
  share-alike (`SA`).
* The disagreement is enough to block Robot SF from committing derived maps, extracted trajectories,
  converted fixtures, or redistributed asset subsets.

## Robot SF Policy

Allowed now:

* Context notes that cite source URLs, upstream commit SHA, access date, observed license text, and
  the conservative decision.
* Metadata-only manifests with upstream file names, upstream blob SHAs, source URLs, and retrieval
  commands, provided they do not embed raw dataset contents or make the manifest a durable runtime
  dependency.
* Conceptual references to scenario categories, semantic-map availability, sensor layout, and
  evaluation tasks.

Not allowed now:

* Raw dataset files, camera images, point clouds, annotations, semantic-map PNGs, or semantic-map
  JSON files.
* Derived Robot SF maps, converted map fixtures, sampled trajectories, simplified occupancy grids,
  or bundled checksums that downstream benchmark jobs require as a durable dependency.
* Benchmark-strength claims whose execution depends on SiT assets.

## Smallest Future Artifact

If maintainers later want a reviewable staging step, the smallest safe artifact is a metadata-only
manifest for the upstream `semantic-maps/` directory:

* upstream repository URL,
* pinned commit SHA,
* file names and GitHub blob SHAs only,
* license conflict note,
* a retrieval command that fetches from upstream only after the user independently accepts the
  current dataset terms.

Minimum validation for that future manifest:

```bash
python -m json.tool docs/context/evidence/<future_sit_manifest>.json >/dev/null
```

No SiT data should be stored under `output/`, `maps/`, `model/`, or tracked fixtures as part of this
issue.
