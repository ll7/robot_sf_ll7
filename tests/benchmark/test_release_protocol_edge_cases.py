"""Edge-case coverage for the v0.2 benchmark release contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.benchmark import release_protocol


def _v02_payload() -> dict[str, Any]:
    """Build a complete v0.2 payload from the pinned predecessor fixture."""
    source_path = Path(
        "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml"
    )
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    loaded = release_protocol.load_release_manifest(source_path)
    payload.update(
        {
            "schema_version": release_protocol.RELEASE_MANIFEST_SCHEMA_VERSION_V0_2,
            "latest_main_base_commit": "a" * 40,
            "matrix": {"expected_episode_cells": 20160, "horizon_steps": 600},
            "publication": {
                "channel": "direct_zenodo_benchmark_dataset",
                "concept_doi": "10.5281/zenodo.99999990",
                "version_doi": "10.5281/zenodo.99999991",
            },
        }
    )
    payload["canonical_campaign_config"] = str(loaded.canonical_campaign_config_path)
    payload["scenario"]["matrix_path"] = str(loaded.scenario_matrix_path)
    suite_policy = Path(
        "configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1_suite_policy.yaml"
    ).resolve()
    route_certification = Path(
        "configs/benchmarks/route_clearance_certifications_v1.yaml"
    ).resolve()
    payload["scenario"].update(
        {
            "suite_policy_path": str(suite_policy),
            "suite_policy_sha256": release_protocol._sha256_file(suite_policy),
            "route_certification_path": str(route_certification),
            "route_certification_sha256": release_protocol._sha256_file(route_certification),
        }
    )
    seed_sets = Path("configs/benchmarks/seed_sets_v1.yaml").resolve()
    payload["seed_policy"].update(
        {
            "seed_sets_path": str(seed_sets),
            "seed_sets_sha256": release_protocol._sha256_file(seed_sets),
            "resolved_seeds": list(range(111, 141)),
        }
    )
    payload["metrics"]["snqi_weights_path"] = str(loaded.snqi_weights_path)
    payload["metrics"]["snqi_baseline_path"] = str(loaded.snqi_baseline_path)
    payload["metrics"]["snqi_claim_policy"] = "advisory_no_ranking"
    payload["provenance"]["doi"] = "10.5281/zenodo.99999991"
    payload["citation_path"] = str(loaded.citation_path)
    payload["release_checklist_path"] = str(loaded.release_checklist_path)
    return payload


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda payload: payload.update(latest_main_base_commit="short"),
            "latest_main_base_commit",
        ),
        (lambda payload: payload.update(matrix={}), "matrix.expected_episode_cells"),
        (lambda payload: payload["matrix"].pop("horizon_steps"), "matrix.horizon_steps"),
        (lambda payload: payload.update(publication=[]), "publication must be a mapping"),
        (
            lambda payload: payload["publication"].update(channel="github_release"),
            "publication.channel",
        ),
        (lambda payload: payload.update(scenario=[]), "scenario must be a mapping"),
        (lambda payload: payload.update(seed_policy=[]), "seed_policy must be a mapping"),
        (
            lambda payload: payload["seed_policy"].update(resolved_seeds=[]),
            "resolved_seeds",
        ),
        (
            lambda payload: payload["metrics"].update(snqi_claim_policy="ranking"),
            "snqi_claim_policy",
        ),
        *(
            (
                lambda payload, doi=doi: payload["publication"].update(concept_doi=doi),
                "fresh Zenodo concept",
            )
            for doi in sorted(release_protocol.HISTORICAL_ZENODO_CONCEPT_DOIS)
        ),
        (
            lambda payload: payload["publication"].update(version_doi="10.5281/zenodo.19563812"),
            "reserved Zenodo version",
        ),
        (
            lambda payload: payload["publication"].update(version_doi="not-a-doi"),
            "reserved Zenodo version",
        ),
    ],
)
def test_v02_loader_rejects_each_incomplete_publication_contract(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    """v0.2 loading fails closed before a malformed contract can run."""
    payload = _v02_payload()
    mutation(payload)
    path = tmp_path / "release-v0.2.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        release_protocol.load_release_manifest(path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("suite_policy_sha256", "0" * 64, "suite_policy_sha256"),
        ("route_certification_sha256", "0" * 64, "route_certification_sha256"),
        ("seed_sets_sha256", "0" * 64, "seed_sets_sha256"),
        ("resolved_seeds", (999,), "resolved_seeds"),
        ("expected_episode_cells", 1, "expected_episode_cells"),
        ("expected_horizon_steps", 500, "matrix.horizon_steps"),
        ("doi", "10.5281/zenodo.other", "provenance.doi"),
        ("concept_doi", "10.5281/zenodo.19563812", "fresh Zenodo concept"),
        ("version_doi", "10.5281/zenodo.19482025", "fresh Zenodo version"),
        ("concept_doi", "10.5281/zenodo.99999991", "concept and version DOI"),
    ],
)
def test_v02_validation_reports_pinned_asset_and_identity_drift(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    """Loaded v0.2 manifests report post-load provenance and matrix drift."""
    payload = _v02_payload()
    path = tmp_path / "release-v0.2.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    manifest = release_protocol.load_release_manifest(path)
    cfg = release_protocol.load_campaign_config(manifest.canonical_campaign_config_path)
    drifted = release_protocol.BenchmarkReleaseManifest(**{**manifest.__dict__, field: value})
    report = release_protocol.validate_release_manifest(drifted, campaign_config=cfg)
    assert report["status"] == "invalid"
    assert any(match in problem for problem in report["problems"])
